"""
judge_jmq_fixed.py — test correct Zen endpoint for muse-spark
Endpoint per https://opencode.ai/docs/zen/: https://opencode.ai/zen/v1/responses, model muse-spark-1.3-contributor-free
Uses Responses API (OpenAI Responses, not chat/completions).
"""
import json, urllib.request
from pathlib import Path
ROOT = Path("/root/AblationWriting")
KEY = (ROOT / ".opencode_key").read_text().strip()
# try Responses API
for model in ["muse-spark-1.3-contributor-free", "muse-spark-1.3"]:
    for base in ["https://opencode.ai/zen/v1/responses"]:
        print(f"TRY model={model} url={base}", flush=True)
        payload = {"model": model, "input": "Say OK in JSON {\"ok\": true}"}
        data = json.dumps(payload).encode()
        try:
            req = urllib.request.Request(base, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode()
                print(f"OK {resp.status} {body[:2000]}", flush=True)
        except Exception as e:
            import traceback
            print(f"FAIL {e}", flush=True)
            try:
                # try to read HTTPError body
                import urllib.error
                if isinstance(e, urllib.error.HTTPError):
                    print(e.read().decode()[:2000], flush=True)
            except Exception as e2:
                print(f"no body {e2}", flush=True)
