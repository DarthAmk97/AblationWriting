import fitz
d = fitz.open("paper/main.pdf")
full = "\n".join(p.get_text() for p in d)
for probe in ["ANCHOR-CONE", "+0.193", "LEX-MATCH", "0.343", "Holm", "ablated win", "preservation", "whiten"]:
    print(probe, full.count(probe))