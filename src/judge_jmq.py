"""
judge_jmq.py — DEFT-compatible blind pairwise judge via opencode Zen muse-spark
CORRECT per https://opencode.ai/docs/zen/ (verified 2026-09-07):
- endpoint https://opencode.ai/zen/v1/responses (Responses API, NOT chat/completions)
- model muse-spark-1.3-contributor-free (free contributor tier; paid muse-spark-1.3 also available)
- auth Bearer <OPENCODE_ZEN_KEY>
- VAST datacenter IP blocked by Cloudflare 1010 → run judging OFF-VAST (local), generations synced from VAST.
Protocol Sec 12.5: blind, randomized order, separate calls per dimension.
"""
import os, json, random, time, hashlib
from pathlib import Path
ROOT = Path("/root/AblationWriting")
# key location: /root/AblationWriting/.opencode_key on VAST, or env locally
def get_key():
    p = ROOT / ".opencode_key"
    if p.exists():
        return p.read_text().strip()
    # local fallback: X:/AblationWriting/.opencode_key_local? Use env
    return os.environ.get("OPENCODE_API_KEY", "")
MODEL = os.environ.get("OPENCODE_MODEL", "muse-spark-1.3-contributor-free")
BASE = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1/responses")

OVERALL_INSTR = "You are a blind writing-quality judge. Choose better writing based on quality and prompt-following. No model identities. Output JSON {\"winner\": \"A\"|\"B\"|\"tie\", \"reason\": \"...\"}."
DIM_INSTR = {
    "clarity": "Judge clarity only. Output JSON {\"winner\":...}.",
    "coherence": "Judge coherence only. Output JSON {\"winner\":...}.",
    "creativity": "Judge creativity only. Output JSON {\"winner\":...}.",
    "depth": "Judge depth/meaningful detail only. Output JSON {\"winner\":...}.",
    "relevance": "Judge prompt relevance only. Output JSON {\"winner\":...}.",
}

def call_responses(instructions, user_text, max_retries=3):
    import urllib.request
    KEY = get_key()
    payload = {"model": MODEL, "instructions": instructions, "input": user_text, "temperature": 0}
    data = json.dumps(payload).encode()
    last = ""
    req_id = ""
    for attempt in range(max_retries):
        req_id = hashlib.sha256(f"{time.time()}-{attempt}-{user_text[:50]}".encode()).hexdigest()[:16]
        try:
            req = urllib.request.Request(BASE, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode()
                j = json.loads(body)
                # Responses API: output[1].content[0].text (message)
                txt = ""
                for o in j.get("output", []):
                    if o.get("type") == "message":
                        for c in o.get("content", []):
                            if c.get("type") == "output_text":
                                txt += c.get("text", "")
                return {"ok": True, "raw": txt, "request_id": j.get("id", req_id), "resp_id": j.get("id"), "retries": attempt, "usage": j.get("usage")}
        except Exception as e:
            last = str(e)
            try:
                import urllib.error
                if isinstance(e, urllib.error.HTTPError):
                    last += " body:" + e.read().decode()[:1000]
            except:
                pass
            time.sleep(2*(attempt+1))
    return {"ok": False, "raw": "", "request_id": req_id, "error": last, "retries": max_retries}

def judge_pair(prompt, a, b, instructions=OVERALL_INSTR):
    flip = random.Random(int(hashlib.sha256((prompt+a[:100]+b[:100]).encode()).hexdigest()[:8],16)).random() < 0.5
    if not flip:
        pa, pb = a, b
    else:
        pa, pb = b, a
    user = f"Prompt:\n{prompt}\n\nCandidate A:\n{pa[:4000]}\n\nCandidate B:\n{pb[:4000]}\n\nChoose better writing. Output JSON."
    res = call_responses(instructions, user)
    # parse
    winner_presented = None
    try:
        txt = res.get("raw","").strip()
        # extract JSON
        start = txt.find("{")
        end = txt.rfind("}")+1
        if start >= 0 and end > start:
            j = json.loads(txt[start:end])
            winner_presented = j.get("winner")
    except:
        pass
    if winner_presented == "A":
        winner_original = ("b" if flip else "a")
    elif winner_presented == "B":
        winner_original = ("a" if flip else "b")
    elif winner_presented == "tie":
        winner_original = "tie"
    else:
        winner_original = "unparsed"
    return {"presented_flip": flip, "winner_presented": winner_presented, "winner_original": winner_original, **res}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        print(f"BASE={BASE} MODEL={MODEL}", flush=True)
        r = judge_pair("Write a haiku about the sea.", "Waves crash on the shore, salty wind sings.", "The sea is water that is wet and blue and has waves and stuff.")
        print(json.dumps(r, indent=2)[:4000])
