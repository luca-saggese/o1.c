#!/usr/bin/env python3
"""
Dump the frozen Python layout oracle for the M1-post native layout unit test.

Emits, to a flat text file parsed by tests/unit/layout_pipe.c:

  1. The absolute bboxes produced by parse_layout_bboxes for a fixed sample
     layout JSON (in the order they are drawn by draw_bbox_layout, i.e. top-5
     by area desc), with the color assigned to each.
  2. The full output of create_layout_reference_images for 1-2 synthetic
     reference images (ref_max_size=None so no resize is involved): each
     bordered ref plus the black layout canvas, as float32 RGB row-major.

Usage:  python dump_layout_oracle.py <out.txt>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "python"))

from models.utils import (
    parse_layout_bboxes,
    draw_bbox_layout,
    create_layout_reference_images,
    DEFAULT_COLORS,
    MAX_BOX,
)

IMAGE_W, IMAGE_H = 256, 192

# Fixed sample layout JSON: dict-style boxes (some with 0-100 coords), a bare
# 4-element list, and a "text"/"label" variant.
LAYOUT_JSON = (
    '['
    '{"bbox":[0.1,0.6,0.1,0.4],"text":"a"},'
    '{"bbox":[50,90,20,70],"label":"b"},'
    '{"bbox":[0.7,0.95,0.6,0.9]},'
    '[0.2,0.4,0.7,0.95],'
    '{"bbox":[0.5,0.55,0.5,0.6],"text":"c"}'
    ']'
)


def synth_ref(w, h):
    """Deterministic integer-valued RGB pattern; identical formula in C."""
    out = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            out[idx + 0] = (x * 3 + y) % 256
            out[idx + 1] = (y * 5 + x * 2) % 256
            out[idx + 2] = (x * y) % 256
    return Image.frombytes("RGB", (w, h), bytes(out))


import json
from PIL import Image  # noqa: E402


def dump(out_path):
    layout_data = json.loads(LAYOUT_JSON)

    lines = []

    # ---- 1. parse + draw_bbox_layout colors ----
    parsed = parse_layout_bboxes(layout_data, IMAGE_W, IMAGE_H)
    layout_img, color_list = draw_bbox_layout(
        parsed, image_width=IMAGE_W, image_height=IMAGE_H, return_color=True
    )
    lines.append(f"WIDTH {IMAGE_W}")
    lines.append(f"HEIGHT {IMAGE_H}")
    lines.append(f"NBOX {len(parsed)}")

    # The colors/order that draw_bbox_layout uses (top MAX_BOX by area desc).
    sorted_bboxes = sorted(parsed, key=lambda it: _area(it), reverse=True)[:MAX_BOX]
    # Reconstruct which color goes to which drawn box (they map via orig_idx).
    for s_idx, item in enumerate(sorted_bboxes):
        x1, y1, x2, y2 = item["bbox"]
        color = DEFAULT_COLORS[s_idx % len(DEFAULT_COLORS)]
        lines.append(
            f"BOX {s_idx} {x1} {y1} {x2} {y2} "
            f"{color[0]} {color[1]} {color[2]}"
        )

    # ---- 2. create_layout_reference_images (no resize) ----
    refs = [synth_ref(40, 30), synth_ref(30, 40)]
    outputs = create_layout_reference_images(
        ref_pils=refs,
        layout_bboxes=layout_data,
        image_width=IMAGE_W,
        image_height=IMAGE_H,
        ref_max_size=None,
        patch_size=32,
    )

    lines.append(f"NREF {len(refs)}")
    lines.append(f"NOUT {len(outputs)}")
    for idx, im in enumerate(outputs):
        w, h = im.size
        lines.append(f"IMG {idx} {w} {h}")
        px = im.convert("RGB")
        data = px.tobytes()  # RGB bytes, row-major
        # float32 values per pixel channel
        vals = []
        for ch in data:
            vals.append("%.6f" % (ch / 255.0))
        for r in range(h):
            row = vals[r * w * 3:(r + 1) * w * 3]
            lines.append(" ".join(row))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _area(item):
    x1, y1, x2, y2 = item["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    dump(sys.argv[1])
