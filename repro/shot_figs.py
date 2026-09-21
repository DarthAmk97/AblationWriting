import fitz
d = fitz.open("paper/main.pdf")
for i, p in enumerate(d):
    t = p.get_text()
    if "Every freeze in the" in t or "Rank choice is empirical" in t:
        pix = p.get_pixmap(dpi=80)
        pix.save("repro/shot_figs.png")
        print("figs-page", i)
        break