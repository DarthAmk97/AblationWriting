import fitz
d = fitz.open("paper/main.pdf")
full = "\n".join(p.get_text() for p in d)
for probe in ["RAND-RANK-DOSE", "0.0005", "degenerate", "VAL1 L2-1", "Bilateral excluded", "factual drift", "0.520", "Rank response"]:
    print(probe, full.count(probe))