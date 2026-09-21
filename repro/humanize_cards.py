import shutil, os
HUMAN = {"L1": "Llama-3.2-1B", "O1": "OLMo-2 1B", "Q08": "Qwen3.5-0.8B", "Q20": "Qwen3.5-2B", "G2": "Gemma-4-E2B"}
SIBS = {
 "L1": ["O1", "Q08", "Q20", "G2"],
 "O1": ["L1", "Q08", "Q20", "G2"],
 "Q08": ["L1", "O1", "Q20", "G2"],
 "Q20": ["L1", "O1", "Q08", "G2"],
 "G2": ["L1", "O1", "Q08", "Q20"],
}
CODELINE = "Pipeline codename %s: internal shorthand from the eight-model sweep, safe to ignore."
for m in HUMAN:
    os.makedirs("hf_cards_v5/" + m, exist_ok=True)
    shutil.copy("hf_cards_v4/" + m + "/README.md", "hf_cards_v5/" + m + "/README.md")
def sub(path, pairs):
    t = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert t.count(old) == 1, (path, old[:70], t.count(old))
        t = t.replace(old, new)
    open(path, "w", encoding="utf-8").write(t)
    print("edited", path, len(pairs))
for m, h in HUMAN.items():
    p = "hf_cards_v5/" + m + "/README.md"
    sibs_plain = ", ".join([HUMAN[s] + " cone" for s in SIBS[m]])
    pairs = [
        (" (H2 safety-cone ablation)", " (frozen safety-cone ablation)"),
        (CODELINE % m, "In file names and code this model is keyed " + m + "; everywhere else this card names it " + h + "."),
        ("Siblings: " + ", ".join([s + "-cone" for s in SIBS[m]]) + " in the same collection.",
         "Sibling cards: " + sibs_plain + ", in the same collection."),
    ]
    for s in SIBS[m]:
        pairs.append(("[" + s + "-cone](", "[" + HUMAN[s] + " cone]("))
    sub(p, pairs)
print("ALL-CARDS-DONE")