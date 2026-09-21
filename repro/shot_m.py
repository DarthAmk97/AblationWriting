import fitz
d = fitz.open("paper/main.pdf")
for i, p in enumerate(d):
    t = p.get_text()
    if "5\tMethodology" in t or "\nMethodology\n" in t:
        p.get_pixmap(dpi=80).save("repro/shot_method.png")
        print("method-page", i)
        break