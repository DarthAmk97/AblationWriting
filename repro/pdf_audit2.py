import fitz
d = fitz.open("paper/main.pdf")
nimg = sum(len(p.get_images()) for p in d)
print("images:", nimg)
print("toc:", [t for t in d.get_toc()][:30])