#!/usr/bin/env python3
"""
rotateimage.py
─────────────────────────────────────────────────────────────
Rotate the input image 90° counter-clockwise and save it as
<original_name>.rotated.<ext> in the same folder.

usage:
    python rotateimage.py page.png
"""

import cv2
import sys
from pathlib import Path

def rotate_ccw(img_bgr):
    """Return the image turned 90° counter-clockwise."""
    return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

def main():
    if len(sys.argv) != 2:
        print("usage: python rotateimage.py IMAGE_FILE")
        sys.exit(1)

    src = Path(sys.argv[1])
    img = cv2.imread(str(src))
    if img is None:
        sys.exit(f"✖ can’t read {src}")

    rotated = rotate_ccw(img)

    out_path = src.with_stem(src.stem + ".rotated")
    cv2.imwrite(str(out_path), rotated)
    print(f"✓ saved {out_path}")

if __name__ == "__main__":
    main()
