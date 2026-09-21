import sys
HUMAN = {
 "L1": "Llama-3.2-1B",
 "O1": "OLMo-2 1B",
 "Q08": "Qwen3.5-0.8B",
 "Q20": "Qwen3.5-2B",
 "G2": "Gemma-4-E2B",
}
BASE = {
 "L1": "Llama-3.2-1B-Instruct",
 "O1": "OLMo-2-0425-1B-Instruct",
 "Q08": "Qwen3.5-0.8B",
 "Q20": "Qwen3.5-2B",
 "G2": "Gemma-4-E2B-it",
}
def sub(path, pairs):
    t = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert t.count(old) == 1, (path, old[:60], t.count(old))
        t = t.replace(old, new)
    open(path, "w", encoding="utf-8").write(t)
    print("edited", path, len(pairs))