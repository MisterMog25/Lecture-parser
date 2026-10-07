#!/usr/bin/env python3
"""Crop Webex meeting chrome off a screen capture, leaving just the shared content.

A 1280x720 Webex capture has the shared window at ~(192,190)-(1092,641); the rest
is meeting chrome (header, participant video, bottom toolbar). This crops that
window and scales the box for other resolutions.

Usage:
    python3 tools/crop_webex.py <image> [<image> ...]
    # writes <name>_crop.jpg next to each input
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# shared window inside a 1280x720 capture
BOX = (192, 190, 1088, 641)
REF = (1280, 720)


def crop(path: Path) -> Path:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    sx, sy = w / REF[0], h / REF[1]
    box = (round(BOX[0] * sx), round(BOX[1] * sy), round(BOX[2] * sx), round(BOX[3] * sy))
    out = path.with_name(path.stem + "_crop.jpg")
    im.crop(box).save(out, quality=90)
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        p = Path(arg)
        if not p.is_file():
            print(f"skip (not a file): {p}", file=sys.stderr)
            continue
        print(crop(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
