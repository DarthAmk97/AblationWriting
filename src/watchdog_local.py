"""Local HF/JMQ health report and conservative resumable JMQ repair."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "dump_mirror/judging/inputs/test_jmq_pairs.parquet"
SHARDS = 4
VALID_WINNERS = {"a", "b", "tie"}
MMD_MODELS = ("L1", "O1", "Q08", "Q20", "G2")


def valid_results(data):
    return data["ok"].fillna(False) & data["winner"].astype(str).str.lower().isin(VALID_WINNERS)


def processes(name, pattern):
    query = (
        f"Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq '{name}' -and "
        f"$_.CommandLine -match '{pattern}' }} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", query], text=True, timeout=30).strip()
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def created_ms(proc):
    match = re.search(r"Date\((\d+)", str(proc.get("CreationDate", "")))
    return int(match.group(1)) if match else 0


def shard_from(proc):
    match = re.search(r"--shard\s+(\d+)", proc.get("CommandLine") or "")
    return int(match.group(1)) if match else None


def checkpoint(shard, assigned_ids, all_done):
    path = ROOT / f"judgments/jmq_overall_s{shard}.parquet"
    if not path.exists():
        return len(all_done & assigned_ids), None
    return len(all_done & assigned_ids), (time.time() - path.stat().st_mtime) / 60


def valid_mmd(geometry, model):
    try:
        metrics_path = geometry / model / "mmd_metrics.json"
        done = json.loads((geometry / model / "mmd_done.json").read_text())
        metrics = json.loads(metrics_path.read_text())
        if done.get("metrics_sha256") != hashlib.sha256(metrics_path.read_bytes()).hexdigest():
            return False
        if metrics.get("model") != model or done.get("model") != model:
            return False
        for split, documents, rows in (("val2", 512, 1536), ("test", 2000, 6000)):
            if metrics.get("splits", {}).get(split, {}).get("n") != documents:
                return False
            if done.get("n", {}).get(split) != documents:
                return False
            if pq.ParquetFile(geometry / model / f"mmd_embeddings_{split}.parquet").metadata.num_rows != rows:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def launch(shard):
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / "local_jmq.py"), "--inputs", str(FEED),
           "--out", str(ROOT / f"judgments/jmq_overall_s{shard}.parquet"),
           "--dim", "overall", "--shard", str(shard), "--num_shards", str(SHARDS)]
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    env = os.environ.copy()
    env["OPENCODE_MODEL"] = "opencode-go/muse-spark-1.3-contributor"
    env["OPENCODE_USE_CLI"] = "1"
    with open(logs / f"jmq_shard{shard}.log", "a") as out, open(logs / f"jmq_shard{shard}.err.log", "a") as err:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err, env=env,
                                creationflags=flags, close_fds=True)
    return proc.pid


def launched_target(shard):
    path = ROOT / f"logs/jmq_shard{shard}.log"
    if not path.exists():
        return None
    matches = re.findall(r"of\s+(\d+)\s+assigned", path.read_text(errors="replace"))
    return int(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-jmq", action="store_true")
    args = parser.parse_args()
    now_ms = int(time.time() * 1000)
    feed = pd.read_parquet(FEED) if FEED.exists() else pd.DataFrame()
    ids = feed["pair_id"].astype(str).tolist() if "pair_id" in feed else []
    assigned = [set(ids[s::SHARDS]) for s in range(SHARDS)]
    all_done = set()
    for path in (ROOT / "judgments").glob("jmq_overall_s*.parquet"):
        try:
            data = pd.read_parquet(path)
            all_done.update(data.loc[valid_results(data), "pair_id"].astype(str))
        except Exception:
            pass
    workers = processes("python.exe", "local_jmq.py")
    judges = processes("opencode.exe", "Judge the attached candidates")
    downloads = processes("hf.exe", "download") + processes("python.exe", "huggingface_cli|hf download")
    by_shard = {shard_from(proc): proc for proc in workers if shard_from(proc) is not None}
    stale_parents = {p["ParentProcessId"] for p in judges if created_ms(p) and now_ms - created_ms(p) > 7 * 60 * 1000}
    actions = []
    status = []
    for shard in range(SHARDS):
        complete, age_min = checkpoint(shard, assigned[shard], all_done)
        worker = by_shard.get(shard)
        stale = bool(worker and worker["ProcessId"] in stale_parents)
        if worker and launched_target(shard) not in {None, len(assigned[shard])}:
            stale = True
        worker_age_min = ((now_ms - created_ms(worker)) / 60000
                          if worker and created_ms(worker) else 0)
        if (worker and worker_age_min > 30 and age_min is not None and age_min > 30
                and not any(p["ParentProcessId"] == worker["ProcessId"] for p in judges)):
            stale = True
        if args.repair_jmq and complete < len(assigned[shard]) and (not worker or stale):
            if worker:
                subprocess.run(["taskkill", "/PID", str(worker["ProcessId"]), "/T", "/F"], capture_output=True, text=True)
            pid = launch(shard)
            actions.append(f"restart shard{shard} pid={pid} reason={'stale' if stale else 'missing'}")
            worker = {"ProcessId": pid}
            stale = False
        status.append((shard, complete, len(assigned[shard]), age_min, worker, stale))

    incomplete = any(done < total for _, done, total, _, _, _ in status)
    bad = [s for s in status if s[1] < s[2] and (not s[4] or s[5])]
    verdict = "STALLED" if bad else "DEGRADED" if not FEED.exists() else "HEALTHY"
    manifest = ROOT / "dump_mirror/dump_manifest.json"
    staged = "missing"
    if manifest.exists():
        try:
            staged = json.loads(manifest.read_text()).get("staged_at", "unknown")
        except Exception:
            staged = "unreadable"
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    print(f"LOCAL_VERDICT {verdict} - {time.strftime('%Y-%m-%d %H:%M', time.localtime())}")
    print(f"MIRROR staged_at={staged} feed={len(feed)} downloads={len(downloads)} free={free_gb:.1f}GiB")
    if (ROOT / ".local_ahead.json").exists():
        print("HF_SYNC local_ahead=true policy=stage-separately direct_dump_mirror_download=forbidden")
    for shard, done, total, age_min, worker, stale in status:
        age_text = "none" if age_min is None else f"{age_min:.1f}m"
        call = next((p for p in judges if worker and p["ParentProcessId"] == worker["ProcessId"]), None)
        call_age = (now_ms - created_ms(call)) / 60000 if call and created_ms(call) else None
        print(f"JMQ shard{shard} {done}/{total} pid={worker.get('ProcessId') if worker else '-'} checkpoint_age={age_text} call_age={f'{call_age:.1f}m' if call_age is not None else '-'} stale={stale}")
    merged = ROOT / "judgments/jmq_overall.parquet"
    if not incomplete:
        frames = [pd.read_parquet(ROOT / f"judgments/jmq_overall_s{shard}.parquet") for shard in range(SHARDS)]
        complete = pd.concat(frames, ignore_index=True)
        complete = complete[valid_results(complete)]
        if complete["pair_id"].astype(str).duplicated().any():
            raise RuntimeError("refusing JMQ merge with pair_ids present in multiple shards")
        complete = complete.set_index(complete["pair_id"].astype(str)).loc[ids].reset_index(drop=True)
        if len(complete) != len(feed) or set(complete["pair_id"].astype(str)) != set(ids):
            raise RuntimeError("refusing incomplete JMQ merge")
        unchanged = merged.exists() and pd.read_parquet(merged).equals(complete)
        if not unchanged:
            tmp = merged.with_suffix(".parquet.tmp")
            complete.to_parquet(tmp, index=False)
            tmp.replace(merged)
            actions.append(f"merged {len(complete)} canonical judgments")
    merged_rows = len(pd.read_parquet(merged)) if merged.exists() else 0
    print(f"JMQ saved_unique={len(all_done)} merged={merged_rows} incomplete={incomplete} judge_children={len(judges)}")
    print(f"ACTIONS {actions or 'none'}")
    geometry = ROOT / "artifacts/geometry/H2"
    mmd_done = sum(valid_mmd(geometry, model) for model in MMD_MODELS)
    mmd_embeddings = len(list(geometry.glob("*/mmd_embeddings_*.parquet")))
    controls_done = len(list((ROOT / "dump_mirror/artifacts/geometry/H2").glob("*/controls_done.json")))
    vast_state = "expected-off" if (ROOT / ".vast_intentionally_off").exists() else "required"
    print(f"REPRO core_mirror=verified mmd={mmd_done}/5 embeddings={mmd_embeddings}/10 controls={controls_done}/5 vast={vast_state}")
    combined = ROOT / "metrics/combined_test_jmq.parquet"
    combined_ready = combined.exists() and {"mmd_test_recovery", "r_l2_3"} <= set(pd.read_parquet(combined).columns)
    print("DOWNSTREAM " + " ".join([
        f"stats={'done' if (ROOT / 'metrics/jmq_bootstrap.parquet').exists() and mmd_done == 5 else 'pending'}",
        f"figures={'done' if combined_ready and (ROOT / 'figures/combined_test_jmq.png').exists() and (ROOT / 'figures/combined_test_jmq.svg').exists() else 'pending'}",
        f"paper={'active' if processes('python.exe', 'latex|paper|figure|bootstrap|stats') else 'pending'}",
    ]))


if __name__ == "__main__":
    main()
