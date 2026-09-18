#!/usr/bin/env python3
"""
Train a HiDream-O1 T2I linear LoRA with the pinned musubi-tuner.

This is a thin, opinionated wrapper around musubi-tuner's
hidream_o1_train_network.py (pinned commit 4e7c7149, see musubi.lock). It
fills in the o1.c contract defaults:

  - task: t2i (visual encoder NOT trainable; no conv/I2I targets)
  - network: LoRA (network_module=networks.lora), linear targets only
  - dtype: BF16 (mixed precision, fp8 disabled)
  - output: <out>/<name>.safetensors (musubi LoRA format)

Usage:

  python3 tools/lora/train.py \
      --dataset-config <prepared>/dataset.toml \
      --pretrained-model <hf-model-dir-or-id> \
      --output-dir out --output-name my_lora \
      --network-dim 16 --network-alpha 16 \
      --learning-rate 1e-4 --max-train-epochs 10

The script locates musubi-tuner via MUSUBI_TUNER_DIR (default: the pinned
checkout path recorded in musubi.lock) and runs it as a subprocess so the
training environment stays isolated from the o1.c toolchain.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "musubi.lock"

DEFAULT_MUSUBI_DIR = str(HERE / ".." / ".." / "external" / "musubi-tuner")


def read_lock() -> dict:
    d = {}
    for line in LOCK.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-config", required=True, type=Path)
    ap.add_argument("--pretrained-model", required=True,
                    help="HF model dir or repo id (HiDream-O1-Image-Dev-2604)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--network-dim", type=int, default=16)
    ap.add_argument("--network-alpha", type=float, default=None,
                    help="defaults to --network-dim (musubi default 1 is NOT used)")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-train-epochs", type=int, default=10)
    ap.add_argument("--save-every-n-epochs", type=int, default=1)
    ap.add_argument("--resolution", default="1024,1024")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--musubi-dir", default=os.environ.get("MUSUBI_TUNER_DIR", DEFAULT_MUSUBI_DIR))
    ap.add_argument("--extra", action="append", default=[],
                    help="extra musubi-tuner args (repeatable)")
    args = ap.parse_args()

    lock = read_lock()
    pinned = lock.get("commit", "")
    train_script = Path(args.musubi_dir) / "hidream_o1_train_network.py"
    if not train_script.exists():
        sys.exit(f"error: musubi-tuner not found at {train_script}\n"
                 f"  clone the pinned commit {pinned} there (see musubi.lock)")

    alpha = args.network_alpha if args.network_alpha is not None else float(args.network_dim)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(train_script),
        "--model_type", "dev",
        "--task", "t2i",
        "--dataset_config", str(args.dataset_config),
        "--pretrained_model_name_or_path", args.pretrained_model,
        "--output_dir", str(args.output_dir),
        "--output_name", args.output_name,
        "--network_module", "networks.lora",
        "--network_dim", str(args.network_dim),
        "--network_alpha", str(alpha),
        "--learning_rate", str(args.learning_rate),
        "--max_train_epochs", str(args.max_train_epochs),
        "--save_every_n_epochs", str(args.save_every_n_epochs),
        "--resolution", args.resolution,
        "--train_batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--mixed_precision", "bf16",
        "--no_fp8_scaled",
        "--skip_t2i_visual_dummy",
        "--save_model_as", "safetensors",
        "--save_precision", "bf16",
    ] + args.extra

    print(f"[train] musubi-tuner {pinned} (pinned)")
    print(f"[train] {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()