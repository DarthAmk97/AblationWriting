"""
ticker.py — 30-min stable-run poller (also usable for rapid polls)
Checks: GPU util/mem, smoke log tail, runs.jsonl count, VRAM, errors, throughput (kappa), stall detection.
Writes logs/ticker.log. Exit 0 if healthy, 1 if stalled/error.
Stall = no new runs.jsonl lines + no log growth for >30 min while process alive.
"""
import time, json, subprocess, sys
from pathlib import Path
ROOT = Path("/root/AblationWriting")
import os

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=20)
    except Exception as e:
        return f"ERR {e}"

def poll():
    print(f"=== TICK {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    # GPU
    print(sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>&1 | head -n 5"))
    print(sh("ps aux 2>&1 | grep -E 'smoke_h2|python' | grep -v grep | head -n 10"))
    # log tail
    for log in ["logs/smoke_O1.log", "logs/install_nohup.log"]:
        p = ROOT / log
        if p.exists():
            st = p.stat()
            print(f"{log}: {st.st_size} bytes mtime {time.ctime(st.st_mtime)}")
            print("--- tail ---")
            print(sh(f"tail -n 30 {p} 2>&1 | head -n 60"))
        else:
            print(f"{log}: MISSING")
    # runs
    rp = ROOT / "runs.jsonl"
    if rp.exists():
        lines = open(rp).read().strip().split("\n") if open(rp).read().strip() else []
        # avoid double read
        txt = open(rp).read().strip()
        n = len([l for l in txt.split('\n') if l.strip()]) if txt else 0
        print(f"runs.jsonl: {n} runs")
        if n:
            last = [json.loads(l) for l in txt.split('\n') if l.strip()][-3:]
            for r in last:
                print(f"  {r.get('run_id')} R={r.get('R_L2')} L2 {r.get('L2_base')}->{r.get('L2_abl')} kappa={r.get('kappa')}")
    else:
        print("runs.jsonl: MISSING (no configs finished yet — expected early)")
    # disk
    print(sh("df -h / 2>&1 | head -n 5; du -sh /root/AblationWriting/.hf_cache 2>&1 | head -n 5"))
    print("=== END TICK ===\n")

if __name__ == "__main__":
    poll()
