import fitz
d = fitz.open("paper/main.pdf")
for i, p in enumerate(d):
    t = p.get_text()
    if "Paired model-minus-human" in t:
        p.get_pixmap(dpi=80).save("repro/shot_method.png")
        print("method-page", i)
    if "holds against dose" in t:
        p.get_pixmap(dpi=80).save("repro/shot_controls.png")
        print("controls-page", i)