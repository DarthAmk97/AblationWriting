import fitz
d = fitz.open("paper/main.pdf")
for i, p in enumerate(d):
    t = p.get_text()
    if "Related work" in t and "Single directions" in t:
        pix = p.get_pixmap(dpi=80)
        pix.save("repro/shot_related.png")
        print("related-page", i)
        break