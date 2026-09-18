#!/usr/bin/env python3
"""
Prepare a musubi-tuner dataset for HiDream-O1 T2I LoRA training.

Builds the directory layout musubi-tuner expects for `--dataset_config`:

    <out>/
      images/            *.png (or *.jpg) training images
      captions/          *.txt captions (same basename as the image)
      dataset.toml       dataset config consumed by musubi-tuner

Input: a directory of (image, caption) pairs. Two accepted layouts:

  1. flat:  <in>/*.png + <in>/*.txt  (caption basename == image basename)
  2. pairs: <in>/<name>/image.png + <in>/<name>/caption.txt

The tool copies/links images and writes captions, then emits a dataset.toml
with the resolution and batch size requested. It never trains anything.

Contract: pinned to musubi-tuner 4e7c7149 (see musubi.lock). The dataset
config format follows musubi_tuner.training.dataset (general + subsets).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

TOML_TEMPLATE = """\
[general]
shuffle_caption = false
caption_extension = ".txt"
keep_tokens = 1

[[subsets]]
image_dir = "{images_dir}"
class_tokens = "{class_tokens}"
num_repeats = {num_repeats}
resolution = [{width}, {height}]
batch_size = {batch_size}
"""


def find_pairs(src: Path):
    """Yield (image_path, caption_path_or_None) for every image in src."""
    images = sorted(src.rglob("*.png")) + sorted(src.rglob("*.jpg")) + \
        sorted(src.rglob("*.jpeg")) + sorted(src.rglob("*.webp"))
    for img in images:
        cap = img.with_suffix(".txt")
        if not cap.exists():
            # pairs layout: <name>/caption.txt
            cap = img.parent / "caption.txt"
        yield img, cap if cap.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="input dir: flat (img+txt) or pairs (<name>/image.png+caption.txt)")
    ap.add_argument("--output", required=True, type=Path,
                    help="output dataset dir (images/, captions/, dataset.toml)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-repeats", type=int, default=1)
    ap.add_argument("--class-tokens", default="")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    args = ap.parse_args()

    if not args.input.is_dir():
        sys.exit(f"error: input dir not found: {args.input}")
    images_dir = args.output / "images"
    caps_dir = args.output / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    caps_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img, cap in find_pairs(args.input):
        dst_img = images_dir / img.name
        if args.copy:
            shutil.copy2(img, dst_img)
        else:
            if dst_img.exists():
                dst_img.unlink()
            os.symlink(img.resolve(), dst_img)
        if cap is not None:
            dst_cap = caps_dir / (img.stem + ".txt")
            shutil.copy2(cap, dst_cap)
        n += 1

    if n == 0:
        sys.exit("error: no images found in input dir")

    toml = TOML_TEMPLATE.format(
        images_dir=str(images_dir.resolve()),
        class_tokens=args.class_tokens,
        num_repeats=args.num_repeats,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )
    (args.output / "dataset.toml").write_text(toml)
    print(f"prepared {n} image/caption pairs -> {args.output}")
    print(f"  dataset.toml: {args.width}x{args.height}, batch {args.batch_size}")


if __name__ == "__main__":
    main()