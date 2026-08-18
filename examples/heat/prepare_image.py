#!/usr/bin/env python3
"""Downsample the example photograph to a grayscale PGM the C program reads.

Usage:  python3 prepare_image.py [--height 256] [--input ../IMG_3293.jpeg]

Writes source_<height>.pgm (binary P5, 8-bit).  The default 256-px source is
committed, so running the example does not require Pillow; rerun this only to
change the resolution.
"""
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--input", default="../IMG_3293.jpeg")
args = parser.parse_args()

im = Image.open(args.input).convert("L")
w = round(args.height * im.width / im.height)
im = im.resize((w, args.height), Image.LANCZOS)
out = f"source_{args.height}.pgm"
with open(out, "wb") as f:
    f.write(f"P5\n{im.width} {im.height}\n255\n".encode())
    f.write(im.tobytes())
print(f"wrote {out}: {im.width} x {im.height}")
