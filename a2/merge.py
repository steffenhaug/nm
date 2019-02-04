from PIL import Image
hr1 = Image.open("hires1.png")
hr2 = Image.open("hires2.png")
hr3 = Image.open("hires3.png")
hr4 = Image.open("hires4.png")

# 1 2
# 3 4

assert hr1.size == hr2.size == hr3.size == hr4.size

w, h = hr1.size
print(w, h)

# remove 1-px white border from pyplot
hr1 = hr1.crop((1, 1, w - 1, h - 1))
hr2 = hr2.crop((1, 1, w - 1, h - 1))
hr3 = hr3.crop((1, 1, w - 1, h - 1))
hr4 = hr4.crop((1, 1, w - 1, h - 1))

w, h = hr1.size
print(w, h)

im = Image.new('RGB', (2 * w, 2 * h))

im.paste(hr1, (0, 0))
im.paste(hr2, (w, 0))
im.paste(hr3, (0, h))
im.paste(hr4, (w, h))

im.save("hires.png")
