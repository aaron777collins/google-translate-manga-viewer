#!/usr/bin/env python3
"""
fix_sideways_morph.py
─────────────────────────────────────────────────────────────────────────────
 1. Rotate page 90° CCW  →  sideways English now lies left-to-right.
 2. Morphological dilate + contour find  →  big, *wide* blobs of text.
 3. Filter blobs:   width > height  AND  OCR says there’s an ASCII letter.
 4. For every blob:
        • crop, rotate 90° CCW (makes the patch vertical in helper space)
        • paste it back full-size (no squish), wiping the old region.
 5. Rotate whole canvas 90° CW back to the original orientation.
 ─────────────────────────────────────────────────────────────────────────────
   Result: every English paragraph—no matter how many lines or how much
           spacing—ends up crisp, horizontal, and readable on the
           original manga page.  Japanese SFX / bubbles stay untouched.
"""

import cv2
import numpy as np
import pytesseract
import re
import sys
from pathlib import Path

# --- point to Tesseract on Windows; comment out on Linux/macOS
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ───────────── tweakables ─────────────────────────────────────────
AREA_MIN   = 1000            # blob must be at least this many pixels
KERNEL_W   = 35                # width  of dilate kernel (join chars / lines)
KERNEL_H   =  7                # height of dilate kernel
PAD        = 4                 # px white border around wiped area
ASCII_RE   = re.compile(r"[A-Za-z]")
# ─────────────────────────────────────────────────────────────────

def is_english(text: str) -> bool:
    return bool(ASCII_RE.search(text))

# -----------------------------------------------------------------
def blobs_with_english(helper_bgr):
    """
    Returns a list of bounding boxes (x0,y0,x1,y1) on the helper image
    that are wide, reasonably big, and contain ASCII text.
    """
    gray = cv2.cvtColor(helper_bgr, cv2.COLOR_BGR2GRAY)
    # binary inverse --> letters = white (255) on black background
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # dilate horizontally so separate letters & lines connect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_W, KERNEL_H))
    dil = cv2.dilate(bw, kernel, iterations=2)

    # find external contours
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    good = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < AREA_MIN or w <= h:           # must be “wide”
            continue

        # quick OCR just on that blob to confirm there’s English
        roi = helper_bgr[y:y+h, x:x+w]
        txt = pytesseract.image_to_string(roi, lang="eng",
                                          config="--psm 6")  # assume a block
        if not is_english(txt):
            continue

        good.append((x, y, x+w, y+h))
    return good

# -----------------------------------------------------------------
def fix_sideways(page_bgr):
    H, W = page_bgr.shape[:2]

    # 1️⃣ rotate 90° CCW so sideways English becomes horizontal
    helper = cv2.rotate(page_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2️⃣ find big wide English blobs
    boxes = blobs_with_english(helper)

    # 3️⃣ process each blob
    canvas = helper.copy()
    for x0, y0, x1, y1 in boxes:
        crop = canvas[y0:y1, x0:x1]

        # rotate patch CCW again so it’s vertical in helper space
        # (→ ends up horizontal after final CW rotation)
        patch = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ph, pw = patch.shape[:2]

        # destination (clip if the patch pokes past right/bottom)
        dx0, dy0 = x0, y0
        dx1 = min(dx0 + pw, canvas.shape[1])
        dy1 = min(dy0 + ph, canvas.shape[0])
        patch = patch[:dy1 - dy0, :dx1 - dx0]

        # white-out a slightly larger rectangle
        cv2.rectangle(canvas,
                      (max(0, dx0 - PAD), max(0, dy0 - PAD)),
                      (min(canvas.shape[1], dx1 + PAD),
                       min(canvas.shape[0], dy1 + PAD)),
                      (255, 255, 255), thickness=-1)

        # paste
        canvas[dy0:dy1, dx0:dx1] = patch

    # 4️⃣ rotate helper canvas back to original orientation
    return cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

# -----------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("usage:  python fixpage.py PAGE.png")
        sys.exit(1)

    src = Path(sys.argv[1])
    page = cv2.imread(str(src))
    if page is None:
        sys.exit(f"✖ can’t read {src}")

    fixed = fix_sideways(page)
    out_path = src.with_suffix(".fixed.png")
    cv2.imwrite(str(out_path), fixed)
    print(f"✓ saved  {out_path}")

if __name__ == "__main__":
    main()
