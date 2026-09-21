import fitz
d = fitz.open("paper/main.pdf")
for i, p in enumerate(d):
    t = p.get_text()
    if "0.0280" in t or "VAL1" in t:
        pix = p.get_pixmap(dpi=80)
        pix.save("repro/shot_stages.png")
        print("stages-page", i)
        break
else:
    print("not-found")