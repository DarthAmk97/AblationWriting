import fitz
d = fitz.open("paper/main.pdf")
pix = d[1].get_pixmap(dpi=80)
pix.save("repro/shot_related2.png")
print("ok")