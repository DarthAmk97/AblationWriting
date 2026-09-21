import fitz
d = fitz.open("paper/main.pdf")
print(len(d), "pages")
for i, p in enumerate(d):
    if "Panel A" in p.get_text() or "Falsification bars" in p.get_text():
        print("hit page", i)