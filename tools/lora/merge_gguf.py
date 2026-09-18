#!/usr/bin/env python3
"""Merge one or more LoRA adapters into a BF16 GGUF base (deployment path).

Produces a new GGUF v3 file with 256-byte alignment and provenance metadata.
Untouched tensors are raw-copied; adapted tensors are merged with the
sequential-bf16-v1 contract (BF16(W + multiplier*scale*(up@down)), FP32
compute). The base GGUF is never modified.

Usage:
  python tools/lora/merge_gguf.py \
      --base artifacts/models/hidream-o1-dev-bf16.gguf \
      --lora artifacts/lora/synthetic/synthetic.safetensors:0.8 \
      --output artifacts/models/hidream-o1-dev-synthetic-bf16.gguf \
      --map config/hidream_lora_map_dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
from safetensors import safe_open

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("o1_lora_common", os.path.join(_here, "common.py"))
common = importlib.util.module_from_spec(_spec)
sys.modules["o1_lora_common"] = common
_spec.loader.exec_module(common)
parse_lora_spec = common.parse_lora_spec
sha256_file = common.sha256_file

_spec2 = importlib.util.spec_from_file_location("o1_lora_gguf", os.path.join(_here, "gguf.py"))
gguf = importlib.util.module_from_spec(_spec2)
sys.modules["o1_lora_gguf"] = gguf
_spec2.loader.exec_module(gguf)


def f32_to_bf16_bytes(a: np.ndarray) -> bytes:
    """RNE F32 -> BF16 bytes, matching hd_f32_to_bf16()."""
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16).tobytes()


def load_map(map_path: str) -> dict:
    with open(map_path) as f:
        m = json.load(f)
    return {e["lora_prefix"]: e for e in m["entries"]}


def load_adapters(lora_specs, amap: dict):
    """Returns list of (path, multiplier, {lora_prefix: (down, up, alpha)})."""
    adapters = []
    for path, mult in lora_specs:
        tensors = {}
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        resolved = {}
        for prefix in amap:
            down_key = f"{prefix}.lora_down.weight"
            up_key = f"{prefix}.lora_up.weight"
            if down_key in tensors and up_key in tensors:
                down = tensors[down_key].astype(np.float32)
                up = tensors[up_key].astype(np.float32)
                rank = down.shape[0]
                alpha_key = f"{prefix}.alpha"
                alpha = float(tensors[alpha_key].reshape(-1)[0]) if alpha_key in tensors else float(rank)
                resolved[prefix] = (down, up, alpha, rank)
        adapters.append((path, mult, resolved))
    return adapters


def main() -> None:
    ap = argparse.ArgumentParser(description="merge LoRA into BF16 GGUF")
    ap.add_argument("--base", required=True, help="base BF16 GGUF")
    ap.add_argument("--lora", action="append", required=True, help="adapter[:multiplier]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--map", required=True)
    args = ap.parse_args()

    amap = load_map(args.map)
    lora_specs = [parse_lora_spec(s) for s in args.lora]
    adapters = load_adapters(lora_specs, amap)

    base = gguf.read_gguf(args.base)
    if base.alignment != 256:
        print(f"warning: base alignment {base.alignment} != 256; output uses 256")

    # Pass 1: validate every adapter target resolves to a base tensor.
    base_by_name = {t.name: t for t in base.tensors}
    for path, mult, resolved in adapters:
        for prefix, (down, up, alpha, rank) in resolved.items():
            entry = amap[prefix]
            base_t = base_by_name.get(entry["base_tensor"])
            if base_t is None:
                print(f"error: adapter {path} target {prefix} -> {entry['base_tensor']} not in base GGUF")
                sys.exit(1)
            if base_t.type != gguf.GGML_TYPE_BF16:
                print(f"error: base tensor {entry['base_tensor']} is not BF16")
                sys.exit(1)
            if base_t.dims != [entry["shape"][0], entry["shape"][1]]:
                print(f"error: shape mismatch {entry['base_tensor']} {base_t.dims} vs map {entry['shape']}")
                sys.exit(1)

    # Pass 2: build output descriptors with 256 alignment.
    out_tensors = []
    offset = 0
    for t in base.tensors:
        offset = gguf.align_up(offset, 256)
        nt = gguf.GgufTensor(t.name, t.n_dims, list(t.dims), t.type, offset)
        out_tensors.append(nt)
        offset += t.nbytes

    # Pass 3: write payload.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".gguf", dir=os.path.dirname(args.output) or ".")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            w = gguf._Writer(f)
            w.u32(gguf.GGUF_MAGIC)
            w.u32(gguf.GGUF_VERSION)
            w.u64(len(out_tensors))
            metadata = dict(base.metadata)
            metadata["general.alignment"] = 256
            metadata["o1.format.version"] = 1
            metadata["o1.weights.dtype"] = "bf16"
            metadata["o1.base.sha256"] = sha256_file(args.base)
            metadata["o1.lora.count"] = len(adapters)
            for i, (path, mult, _) in enumerate(adapters):
                metadata[f"o1.lora.{i}.name"] = os.path.basename(path)
                metadata[f"o1.lora.{i}.sha256"] = sha256_file(path)
                metadata[f"o1.lora.{i}.multiplier"] = mult
                metadata[f"o1.lora.{i}.merge"] = "sequential-bf16-v1"
            w.u64(len(metadata))
            for k, v in metadata.items():
                gguf._write_kv(w, k, v)
            for t in out_tensors:
                w.string(t.name)
                w.u32(t.n_dims)
                for d in t.dims:
                    w.u64(d)
                w.u32(t.type)
                w.u64(t.offset)
            w.pad(256)
            data_start = w.pos

            for t in out_tensors:
                if w.pos - data_start != t.offset:
                    raise ValueError(f"offset mismatch for {t.name}")
                payload = gguf.read_tensor_payload(base, t)
                merged = False
                for path, mult, resolved in adapters:
                    # find adapter entry whose gguf_name matches this tensor
                    for prefix, (down, up, alpha, rank) in resolved.items():
                        if amap[prefix]["base_tensor"] == t.name:
                            w_arr = np.frombuffer(payload, dtype=np.uint16).astype(np.float32)
                            # BF16 -> F32, reshape to [out_dim, in_dim]
                            w_f32 = ((w_arr.astype(np.uint32) << 16).view(np.float32)
                                     .reshape(up.shape[0], down.shape[1]))
                            scale = mult * alpha / rank
                            delta = up @ down
                            merged_w = w_f32 + scale * delta
                            payload = f32_to_bf16_bytes(merged_w)
                            merged = True
                            break
                    if merged:
                        break
                w.write(payload)
                w.pad(256)

        # Finalize: revalidate + atomic rename.
        check = gguf.read_gguf(tmp_path)
        if check.tensor_count != base.tensor_count:
            raise ValueError("output tensor count mismatch")
        os.replace(tmp_path, args.output)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"wrote {args.output} ({os.path.getsize(args.output)/1e9:.2f} GB)")
    print(f"  tensors: {base.tensor_count}, alignment: 256")
    print(f"  adapters: {[os.path.basename(p) for p, _, _ in adapters]}")


if __name__ == "__main__":
    main()