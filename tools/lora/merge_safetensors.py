#!/usr/bin/env python3
"""Offline LoRA merge into a safetensors base (reference / debugging).

Processes one tensor at a time: untouched tensors are raw-copied, adapted
tensors are merged as BF16(W + multiplier*scale*(up@down)) with FP32 compute.
Never loads the whole model into host RAM.

Usage:
  python tools/lora/merge_safetensors.py \
      --base models/dev \
      --lora artifacts/lora/synthetic/synthetic.safetensors:0.8 \
      --output artifacts/lora/synthetic/merged.safetensors \
      --map config/hidream_lora_map_dev.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("o1_lora_common", os.path.join(_here, "common.py"))
common = importlib.util.module_from_spec(_spec)
sys.modules["o1_lora_common"] = common
_spec.loader.exec_module(common)
parse_lora_spec = common.parse_lora_spec


def f32_to_bf16_bytes(a: np.ndarray) -> np.ndarray:
    """RNE F32 -> BF16, matching hd_f32_to_bf16()."""
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16)


def load_map(map_path: str) -> dict:
    with open(map_path) as f:
        m = json.load(f)
    return {e["lora_prefix"]: e for e in m["entries"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="merge LoRA into safetensors base")
    ap.add_argument("--base", required=True, help="model dir with safetensors shards")
    ap.add_argument("--lora", action="append", required=True, help="adapter[:multiplier]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--map", required=True)
    args = ap.parse_args()

    amap = load_map(args.map)
    lora_specs = [parse_lora_spec(s) for s in args.lora]

    # Load all adapter tensors into host RAM (adapters are small).
    adapters = []
    for path, mult in lora_specs:
        tensors = {}
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        adapters.append((path, mult, tensors))

    # Stream the base shards.
    shards = sorted(
        p for p in os.listdir(args.base) if p.endswith(".safetensors")
    )
    if not shards:
        print(f"error: no safetensors shards in {args.base}")
        sys.exit(1)

    out_tensors = {}
    for shard in shards:
        shard_path = os.path.join(args.base, shard)
        with safe_open(shard_path, framework="np") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                merged = False
                for path, mult, tensors in adapters:
                    # base tensor name -> lora prefix
                    if not key.endswith(".weight"):
                        continue
                    module = key[: -len(".weight")]
                    lora_prefix = "lora_unet_" + module.replace(".", "_")
                    entry = amap.get(lora_prefix)
                    if entry is None:
                        continue
                    down_key = f"{lora_prefix}.lora_down.weight"
                    up_key = f"{lora_prefix}.lora_up.weight"
                    if down_key not in tensors or up_key not in tensors:
                        continue  # adapter does not cover this target: leave base unchanged
                    down = tensors[down_key].astype(np.float32)
                    up = tensors[up_key].astype(np.float32)
                    rank = down.shape[0]
                    alpha_key = f"{lora_prefix}.alpha"
                    alpha = float(tensors[alpha_key].reshape(-1)[0]) if alpha_key in tensors else float(rank)
                    scale = mult * alpha / rank
                    w = t.astype(np.float32)
                    delta = up @ down
                    merged_w = w + scale * delta
                    t = f32_to_bf16_bytes(merged_w)
                    merged = True
                    break
                if not merged:
                    # preserve original dtype/bytes
                    pass
                out_tensors[key] = t
        print(f"  {shard}: {len([k for k in out_tensors if k.startswith('model.')])} tensors")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(out_tensors, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()