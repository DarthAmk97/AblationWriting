"""Concise, read-only VAST health report for the 30-minute H2 watchdog."""
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/root/AblationWriting")
MODELS = ("L1", "S3", "Q08", "G4", "G2", "O1", "Q20", "LF12")
JOB_RE = re.compile(r"(full_h2|preserve_full|test_headline|h2_controls|mmd_embed|mmd_score)\.py|drive_full\.sh|stage_dump\.py")
ERROR_RE = re.compile(r"out of memory|OutOfMemoryError|CUDA OOM|Traceback|fatal", re.I)


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as exc:
        return f"ERR {exc}"


def age(path):
    return (time.time() - path.stat().st_mtime) / 60 if path.exists() else None


def frozen(mid):
    path = ROOT / "artifacts/geometry/H2" / mid / "frozen.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("frozen") or "REVERTED"
    except Exception:
        return "BROKEN"


def valid_mmd(mid):
    try:
        geom = ROOT / "artifacts/geometry/H2" / mid
        metrics_path = geom / "mmd_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        done = json.loads((geom / "mmd_done.json").read_text())
        return (
            metrics.get("model") == mid
            and done.get("model") == mid
            and done.get("metrics_sha256") == hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            and metrics.get("splits", {}).get("val2", {}).get("n") == 512
            and metrics.get("splits", {}).get("test", {}).get("n") == 2000
        )
    except (OSError, ValueError, TypeError):
        return False


def queue():
    path = ROOT / "src/drive_full.sh"
    if not path.exists():
        return [], []
    text = path.read_text()
    match = re.search(r'^QUEUE="([^"]*)"', text, re.M)
    items = match.group(1).split() if match else []
    blocked = set((ROOT / ".blocked_models").read_text().split()) if (ROOT / ".blocked_models").exists() else set()
    pending = []
    for item in items:
        mid, stage = item.split(":", 1)
        if mid in blocked:
            continue
        geom = ROOT / "artifacts/geometry/H2" / mid
        marker = {
            "screen0": geom / "screen0_windows.json", "s1": geom / "s1_top12.json",
            "s2": geom / "s2_top4.json", "s3": geom / "frozen.json",
            "preserve": geom / "preservation_full.json", "test": geom / "test_done",
        }.get(stage)
        if marker and marker.exists():
            continue
        if stage in {"preserve", "test"} and frozen(mid) in {None, "REVERTED", "BROKEN"}:
            continue
        pending.append(item)
    return items, pending


def active_log(args):
    model = re.search(r"--models\s+([A-Za-z0-9]+)", args)
    mid = model.group(1) if model else ""
    if "test_headline.py" in args:
        return ROOT / f"logs/full_{mid}_test.log"
    if "preserve_full.py" in args:
        return ROOT / f"logs/full_{mid}_preserve.log"
    stage = re.search(r"--stage\s+([A-Za-z0-9]+)", args)
    if "full_h2.py" in args and stage:
        return ROOT / f"logs/full_{mid}_{stage.group(1)}.log"
    if "stage_dump.py" in args:
        return ROOT / "logs/dump_auto.log"
    if "mmd_embed.py" in args:
        return ROOT / "logs/mmd_embed.log"
    if "mmd_score.py" in args:
        return ROOT / "logs/mmd_score.log"
    if "h2_controls.py" in args:
        return ROOT / "logs/controls.log"
    return ROOT / "logs/driver_full.log" if "drive_full.sh" in args else None


def last_progress(path):
    if not path or not path.exists():
        return "no log"
    lines = path.read_text(errors="replace").splitlines()[-120:]
    hits = [line.strip() for line in lines if re.search(r"\b\d+/\d+\b|TEST R_|PUSHED|STAGED|WROTE", line)]
    return (hits[-1] if hits else (lines[-1].strip() if lines else "empty log"))[:180]


def main():
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    ps_text = command(["ps", "-eo", "pid=,etimes=,time=,args="])
    jobs = [line.strip() for line in ps_text.splitlines() if JOB_RE.search(line)]
    gpu_text = command(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu", "--format=csv,noheader,nounits"])
    try:
        util, used, free, temp = [int(x.strip()) for x in gpu_text.splitlines()[0].split(",")]
    except Exception:
        util = used = free = temp = -1
    all_queue, pending = queue()
    driver_alive = any("drive_full.sh" in line for line in jobs)
    work_jobs = [line for line in jobs if "drive_full.sh" not in line and "stage_dump.py" not in line]
    stalled = []
    for line in jobs:
        log = active_log(line)
        minutes = age(log) if log else None
        if line in work_jobs and minutes is not None and minutes > 60 and util < 5:
            stalled.append(f"stale active job ({minutes:.0f}m): {line}")
    if pending and not driver_alive and not work_jobs:
        stalled.append("driver dead with pending queue")

    recent_errors = []
    for log in sorted((ROOT / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        if age(log) > 60:
            continue
        tail = log.read_text(errors="replace")[-40000:]
        matches = ERROR_RE.findall(tail)
        if matches:
            recent_errors.append(f"{log.name}:{matches[-1]}")

    dump = ROOT / "dump/dump_manifest.json"
    dump_age = age(dump)
    disk_text = command(["df", "-P", str(ROOT)])
    disk_pct = -1
    match = re.search(r"(\d+)%", disk_text.splitlines()[-1] if disk_text else "")
    if match:
        disk_pct = int(match.group(1))
    verdict = "STALLED" if stalled else "DEGRADED" if recent_errors or disk_pct >= 90 or (dump_age is not None and dump_age > 8 * 60) else "HEALTHY"

    print(f"REMOTE_VERDICT {verdict} — {now}")
    print(f"GPU util={util}% used={used}MiB free={free}MiB temp={temp}C; disk={disk_pct}%")
    print(f"QUEUE total={len(all_queue)} pending={len(pending)} next={pending[:4] or ['drained/parked']}")
    print("JOBS:")
    if not jobs:
        print("  none")
    for line in jobs:
        log = active_log(line)
        minutes = age(log) if log else None
        print(f"  {line} | log_age={minutes:.1f}m | {last_progress(log)}" if minutes is not None else f"  {line}")
    print("ARTIFACTS:")
    runs_path = ROOT / "runs.jsonl"
    runs = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()] if runs_path.exists() else []
    for mid in MODELS:
        geom = ROOT / "artifacts/geometry/H2" / mid
        stages = {stage: sum(r.get("model") == mid and r.get("stage") == stage for r in runs) for stage in ("s1", "s2", "s3")}
        smoke = sum(r.get("model") == mid and not r.get("stage") and r.get("hypothesis") != "H2-test" for r in runs)
        marks = [name for name, path in (
            ("screen0", geom / "screen0_windows.json"), ("preserve", geom / "preservation_full.json"),
            ("test", geom / "test_done"), ("mmd", geom / "mmd_done.json"),
            ("controls", geom / "controls_done.json")) if path.exists()]
        frozen_id = frozen(mid)
        frozen_row = next((r for r in runs if r.get("run_id") == frozen_id), None)
        val_r = f"{frozen_row.get('R_L2'):+.3f}" if frozen_row and isinstance(frozen_row.get("R_L2"), (int, float)) else "-"
        test_path = geom / "test_metrics.json"
        test_r = "-"
        if test_path.exists():
            try:
                value = json.loads(test_path.read_text()).get("R_L2_1")
                test_r = f"{value:+.3f}" if isinstance(value, (int, float)) else "-"
            except Exception:
                test_r = "BROKEN"
        preserve_path = geom / "preservation_full.json"
        preserve = "-"
        if preserve_path.exists():
            try:
                preserve = str(json.loads(preserve_path.read_text()).get("pass"))
            except Exception:
                preserve = "BROKEN"
        print(f"  {mid}: smoke={smoke} s1={stages['s1']} s2={stages['s2']} s3={stages['s3']} frozen={frozen_id} VAL2_R={val_r} TEST_R={test_r} preserve={preserve} done={','.join(marks) or '-'}")
    embeddings = list((ROOT / "artifacts/geometry/H2").glob("*/mmd_embeddings_*.parquet"))
    scores = sum(valid_mmd(mid) for mid in ("L1", "O1", "Q08", "Q20", "G2"))
    print(f"MMD embeddings={len(embeddings)}/10 verified_models={scores}/5")
    print(f"DUMP age={dump_age / 60:.1f}h" if dump_age is not None else "DUMP missing")
    print(f"RECENT_ERRORS {recent_errors or 'none'}")
    for item in stalled:
        print(f"STALL {item}")


if __name__ == "__main__":
    main()
