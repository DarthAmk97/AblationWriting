import fitz
d = fitz.open("paper/main.pdf")
full = "\n".join(p.get_text() for p in d)
print(len(d), "pages,", len(full), "chars")
for probe in ["Panel A", "Panel D", "Falsification bars", "Rank response", "How far each", "Blind preference up close", "VAL1", "Preservation gates", "Negative results", "Reproducibility map", "References"]:
    print(probe, full.count(probe))