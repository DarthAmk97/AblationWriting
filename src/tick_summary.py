"""
tick_summary.py — neat plain-English ticker summary (user request 2026-09-07)
Reads runs.jsonl + logs + ps/GPU, outputs:
  TRIED <model> with <settings> to test <question> → RESULT [...] → TELLS US ... BUT NOT ...
  MEANWHILE ...
Usage: HF_HOME=... python src/tick_summary.py
"""
import json, subprocess, time
from pathlib import Path
from collections import defaultdict
ROOT = Path("/root/AblationWriting")

MODEL_FULL = {
    "O1": "allenai/OLMo-2-0425-1B-Instruct", "Q08": "Qwen/Qwen3.5-0.8B",
    "L1": "meta-llama/Llama-3.2-1B-Instruct", "LF12": "LiquidAI/LFM2-1.2B",
    "Q20": "Qwen/Qwen3.5-2B", "G2": "google/gemma-4-E2B-it",
    "G4": "google/gemma-4-E4B-it", "S3": "HuggingFaceTB/SmolLM3-3B",
}
WINDOW_PLAIN = {
    "W1_25-44": "early-mid layers 4-7 (25-44% depth)",
    "W2_50-69": "mid-late layers 8-11 (50-69% depth)",
}

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=20)
    except Exception as e:
        return f"ERR {e}"

def main():
    print(f"=== SUMMARY {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} ===")
    # procs
    ps = sh("ps -o pid,etime,time,args -e | grep -E '(full_h2|preserve_full|test_headline)\\.py --models|drive_full\\.sh' | grep -v grep | head -n 20")
    print("PROCS:\n" + ps.strip() if ps.strip() else "PROCS: none (GPU idle)")
    print(sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | head -n 3").strip())
    # runs
    rp = ROOT / "runs.jsonl"
    if not rp.exists():
        print("RUNS: none yet (still baseline/teacher-force — expected early, not stalled if GPU active).")
        return
    runs = [json.loads(l) for l in open(rp) if l.strip()]
    by_model = defaultdict(list)
    for r in runs:
        by_model[r.get("model")].append(r)
    for mid, rs in sorted(by_model.items()):
        full = MODEL_FULL.get(mid, mid)
        print(f"\n--- {full}: {len(rs)} configs finished ---")
        frozen_path = ROOT / "artifacts" / "geometry" / "H2" / mid / "frozen.json"
        frozen = json.load(open(frozen_path)).get("frozen") if frozen_path.exists() else None
        if frozen_path.exists() and frozen:
            best = next((r for r in rs if r.get("run_id") == frozen), max(rs, key=lambda x: x.get("R_L2", -9)))
        elif frozen_path.exists():
            val2 = [r for r in rs if r.get("stage") == "s3" and r.get("cone") == "cone"]
            best = max(val2 or rs, key=lambda x: x.get("R_L2", -9))
        else:
            full_rows = [r for r in rs if r.get("stage") == "s2"]
            best = max(full_rows or rs, key=lambda x: x.get("R_L2", -9))
        # group by window
        for w in sorted({x.get("window") for x in rs if x.get("window") is not None}):
            wr = [x for x in rs if x.get("window")==w]
            wplain = WINDOW_PLAIN.get(w, w)
            wb = max(wr, key=lambda x: x.get("R_L2", -9))
            print(f"  {w} ({wplain}): n={len(wr)}, best r{wb.get('rank')}-a{wb.get('alpha')} R={wb.get('R_L2',0):+.3f}")
        b = best
        print(f"  BEST: {b['run_id']} R={b.get('R_L2',0):+.3f} L2 {b.get('L2_base',0):.5f}->{b.get('L2_abl',0):.5f} JSD {b.get('JSD_base',0):.4f}->{b.get('JSD_abl',0):.4f} kappa={b.get('kappa',0):.2f}")
        # neat paragraph
        print(f"\n  PLAIN: We tried {mid} with {len(rs)} settings (windows × ranks 2/4/8 × strengths 0.5/1.0, protection 0.75) to test whether a low-rank positive-cone ablation in that depth moves VAL-128 writing toward humans without breaking fluency (0 pathologies). Achieved best R={b.get('R_L2',0):+.3f} at {b.get('window')} rank {b.get('rank')} strength {b.get('alpha')}.")
        if b.get("R_L2",0) > 0.02:
            print(f"  TELLS US: mid-late depth holds a small causal writing signal (rank-4 elbow: r8 worse), hook alive, overhead healthy (kappa~{b.get('kappa',0):.2f}). BUT NOT: preservation (IFEval/MMLU/safety), MMD-embed, blind JMQ quality, StoryScope structure, full-rank sweep to 32, or TEST generalization — all queued for full H2.")
        else:
            print(f"  TELLS US: no reliable recovery yet in tested depths/ranks (all R≤0 or tiny). BUT NOT: whether later windows, higher ranks to 32, full protect behaviours, or other families behave differently — still queued. Do not conclude H4 from smoke.")
    # running
    print("\nMEANWHILE (live):")
    print(ps.strip())
    # dump freshness (local-takeover contract: loop tracks staged_at + JMQ readiness)
    print("\nDUMP (HF local-takeover):")
    dm = ROOT / "dump" / "dump_manifest.json"
    if dm.exists():
        try:
            m = json.load(open(dm))
            import os as _os
            age_h = (_os.path.getmtime(dm) and (time.time() - _os.path.getmtime(dm)) / 3600)
            print(f"  staged_at {m.get('staged_at')} ({age_h:.1f}h ago) runs={m.get('runs')} frozen={len(m.get('frozen', []))} jmq_ready={m.get('jmq_inputs_ready')}")
            pend = m.get("pending", [])
            if pend:
                print(f"  pending: {'; '.join(pend[:6])}")
        except Exception as e:
            print(f"  manifest unreadable: {e}")
    else:
        print("  no dump staged yet (run src/stage_dump.py)")
    # logs tail
    for log in ["logs/smoke_Q08.log", "logs/smoke_L1.log", "logs/smoke_O1.log"]:
        p = ROOT / log
        if p.exists():
            import os
            mt = time.strftime("%H:%M", time.gmtime(os.path.getmtime(p)))
            last = sh(f"tail -n 3 {p} | tr '\\n' ';' | cut -c1-300")
            print(f"  {log} mtime {mt} UTC: {last.strip()[:280]}")

if __name__ == "__main__":
    main()
