#!/usr/bin/env python3
"""Inspect and validate a musubi HiDream-O1 LoRA adapter.

Resolves every adapter key against the canonical map, validates shapes and
dtypes, and reports supported/unsupported/unknown entries. In strict mode
any unsupported or unknown target is a hard error.

Usage:
  python tools/lora/inspect.py \
      --lora artifacts/lora/my_subject/checkpoints/my_subject.safetensors \
      --map config/hidream_lora_map_dev.json \
      [--strict]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
from safetensors import safe_open

# Load common.py without adding tools/lora to sys.path (our inspect_lora.py
# would otherwise shadow the stdlib `inspect` module for numpy).
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("o1_lora_common", os.path.join(_here, "common.py"))
common = importlib.util.module_from_spec(_spec)
sys.modules["o1_lora_common"] = common
_spec.loader.exec_module(common)
LoraAdapter = common.LoraAdapter
LoraEntry = common.LoraEntry
sha256_file = common.sha256_file

SUPPORTED_DTYPES = {"float16", "float32", "bfloat16"}


def _dtype_name(dt) -> str:
    return getattr(dt, "name", str(dt))


def load_map(path: str) -> dict:
    with open(path) as f:
        m = json.load(f)
    by_prefix = {}
    for e in m["entries"]:
        by_prefix[e["lora_prefix"]] = e
    return {"profile": m["profile"], "by_prefix": by_prefix, "n_entries": m["n_entries"]}


def inspect(lora_path: str, map_path: str, strict: bool) -> LoraAdapter:
    amap = load_map(map_path)
    adapter = LoraAdapter(path=lora_path, sha256=sha256_file(lora_path))
    adapter.base_profile = amap["profile"]

    with safe_open(lora_path, framework="np") as f:
        keys = list(f.keys())

        # Group keys by module prefix.
        modules: dict[str, dict[str, str]] = {}
        for key in keys:
            if key.endswith(".lora_down.weight"):
                mod = key[: -len(".lora_down.weight")]
                modules.setdefault(mod, {})["down"] = key
            elif key.endswith(".lora_up.weight"):
                mod = key[: -len(".lora_up.weight")]
                modules.setdefault(mod, {})["up"] = key
            elif key.endswith(".alpha"):
                mod = key[: -len(".alpha")]
                modules.setdefault(mod, {})["alpha"] = key
            else:
                adapter.unknown.append(key)

        for mod, parts in sorted(modules.items()):
            if "down" not in parts or "up" not in parts:
                adapter.unsupported.append(f"{mod}: missing up/down pair")
                continue
            down = f.get_tensor(parts["down"])
            up = f.get_tensor(parts["up"])
            if down.ndim != 2 or up.ndim != 2:
                adapter.unsupported.append(f"{mod}: non-linear (down.ndim={down.ndim} up.ndim={up.ndim})")
                continue
            if _dtype_name(down.dtype) not in SUPPORTED_DTYPES or _dtype_name(up.dtype) not in SUPPORTED_DTYPES:
                adapter.unsupported.append(f"{mod}: unsupported dtype {_dtype_name(down.dtype)}/{_dtype_name(up.dtype)}")
                continue

            rank, in_dim = down.shape
            out_dim, rank2 = up.shape
            if rank != rank2:
                adapter.unsupported.append(f"{mod}: rank mismatch down={rank} up={rank2}")
                continue
            if rank <= 0:
                adapter.unsupported.append(f"{mod}: rank 0")
                continue

            # alpha: scalar tensor or absent (defaults to rank).
            alpha = float(rank)
            if "alpha" in parts:
                a = f.get_tensor(parts["alpha"])
                if a.ndim != 0 or a.size != 1:
                    adapter.unsupported.append(f"{mod}: malformed alpha")
                    continue
                alpha = float(a.reshape(-1)[0])
                if not math.isfinite(alpha) or alpha == 0:
                    adapter.unsupported.append(f"{mod}: invalid alpha {alpha}")
                    continue

            # NaN/Inf check on a sample (full scan is expensive for big adapters).
            for name, arr in (("down", down), ("up", up)):
                if not np.isfinite(arr).all():
                    adapter.unsupported.append(f"{mod}: NaN/Inf in {name}")
                    break
            else:
                entry = amap["by_prefix"].get(mod)
                if entry is None:
                    adapter.unknown.append(mod)
                    continue
                if entry["kind"] != "linear":
                    adapter.unsupported.append(f"{mod}: kind {entry['kind']}")
                    continue
                base_shape = entry["shape"]
                if base_shape != [out_dim, in_dim]:
                    adapter.unsupported.append(
                        f"{mod}: shape mismatch base={base_shape} down={down.shape} up={up.shape}"
                    )
                    continue
                adapter.entries.append(
                    LoraEntry(
                        lora_prefix=mod,
                        base_module=entry["base_module"],
                        base_tensor=entry["base_tensor"],
                        kind="linear",
                        rank=rank,
                        in_dim=in_dim,
                        out_dim=out_dim,
                        alpha=alpha,
                        down_key=parts["down"],
                        up_key=parts["up"],
                        alpha_key=parts.get("alpha", ""),
                    )
                )

    return adapter


def report(adapter: LoraAdapter) -> None:
    print(f"LoRA: {os.path.basename(adapter.path)}")
    print(f"format: musubi LoRA (sha256 {adapter.sha256[:16]}...)")
    print(f"task compatibility: T2I linear")
    print(f"base profile: {adapter.base_profile}")
    print(f"modules: {adapter.supported_count + len(adapter.unsupported) + len(adapter.unknown)}")
    print(f"rank distribution: {adapter.rank_distribution}")
    print(f"alpha distribution: {adapter.alpha_distribution}")
    print()
    print("supported:")
    for e in adapter.entries:
        print(f"  {e.lora_prefix} rank={e.rank} [{e.out_dim},{e.in_dim}] alpha={e.alpha}")
    if adapter.unsupported:
        print("unsupported:")
        for u in adapter.unsupported:
            print(f"  {u}")
    if adapter.unknown:
        print("unknown:")
        for u in adapter.unknown:
            print(f"  {u}")
    print()
    print(f"shape validation: {'PASS' if not adapter.unsupported else 'FAIL'}")
    print(f"mapping validation: {'PASS' if not adapter.unknown else 'FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="inspect musubi LoRA adapter")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    adapter = inspect(args.lora, args.map, args.strict)
    report(adapter)

    if args.strict and (adapter.unsupported or adapter.unknown):
        print("STRICT: unsupported or unknown targets present")
        sys.exit(1)
    if not adapter.entries:
        print("error: no supported linear entries")
        sys.exit(1)


if __name__ == "__main__":
    main()