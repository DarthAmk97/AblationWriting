import fitz
d = fitz.open("paper/main.pdf")
for i in [11, 19, 20]:
    p = d[i]
    pix = p.get_pixmap(dpi=80)
    pix.save("repro/shot_p%d.png" % i)
print("shots-ok")