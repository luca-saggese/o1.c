#!/usr/bin/env python3
"""
hidream_convert.py — safetensors -> GGUF v3 materialized pack for o1.c.

Converts the frozen HiDream safetensors shards into a single GGUF v3 file
with:
  - BF16 tensors (F32 weights cast to BF16, matching the runtime contract)
  - general.alignment = 256 (cuBLAS/cuDNN-friendly)
  - production ordering (embed, block.0 q/k/v/o/gate/up/down, ..., final)
  - rich hidream.* metadata (profile, revision, layout version, checksum)

Usage:
  python3 tools/hidream_convert.py \
      --source models/dev \
      --output artifacts/models/hidream-o1-dev-bf16.gguf \
      --profile dev \
      --revision b6acc2fe452b3120430620dc4354fa442ee081ea

The output is a plain GGUF v3 file readable by the minimal C reader in
src/io/gguf.c (no ggml/llama.cpp dependency).
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
ALIGNMENT = 256

# ggml_type enum (subset we emit)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

# gguf_metadata_value_type
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def f32_to_bf16(f):
    """Round-to-nearest-even F32 -> BF16, matching hd_f32_to_bf16()."""
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    lsb = (u >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    u += rounding_bias
    return (u >> 16) & 0xFFFF


def f32_buf_to_bf16_np(raw):
    """Vectorized F32 bytes -> BF16 bytes using numpy (RNE rounding)."""
    a = np.frombuffer(raw, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    hi = (u >> 16).astype(np.uint16)
    return hi.tobytes()


def align_up(v, a):
    return (v + a - 1) & ~(a - 1)


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.pos = 0

    def write(self, data):
        self.f.write(data)
        self.pos += len(data)

    def pad(self, alignment):
        pad = (alignment - (self.pos % alignment)) % alignment
        if pad:
            self.write(b"\x00" * pad)
        return pad

    def u8(self, v): self.write(struct.pack("<B", v))
    def u16(self, v): self.write(struct.pack("<H", v))
    def u32(self, v): self.write(struct.pack("<I", v))
    def u64(self, v): self.write(struct.pack("<Q", v))
    def i64(self, v): self.write(struct.pack("<q", v))
    def f32(self, v): self.write(struct.pack("<f", v))
    def f64(self, v): self.write(struct.pack("<d", v))

    def string(self, s):
        b = s.encode("utf-8")
        self.u64(len(b))
        self.write(b)

    def kv_string(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_STRING)
        self.string(value)

    def kv_uint32(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_UINT32)
        self.u32(value)

    def kv_uint64(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_UINT64)
        self.u64(value)

    def kv_bool(self, key, value):
        self.string(key)
        self.u32(GGUF_TYPE_BOOL)
        self.u8(1 if value else 0)

    def close(self):
        self.f.close()


def load_shard_index(source_dir):
    """Returns {tensor_name: (shard_path, data_begin, nbytes, dtype, shape)}."""
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_headers = {}
    tensors = {}
    for name, shard in weight_map.items():
        if shard not in shard_headers:
            path = os.path.join(source_dir, shard)
            with open(path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            shard_headers[shard] = (path, 8 + hlen, hdr)
        path, payload_base, hdr = shard_headers[shard]
        t = hdr[name]
        begin, end = t["data_offsets"]
        tensors[name] = {
            "shard": shard,
            "path": path,
            "data_begin": payload_base + begin,
            "nbytes": end - begin,
            "dtype": t["dtype"],
            "shape": t["shape"],
        }
    return tensors


def production_order(tensors):
    """Order tensors for the production device layout: embed, blocks, final."""
    def key(name):
        # model.language_model.layers.N.self_attn.{q,k,v,o}_proj.weight
        # model.language_model.layers.N.mlp.{gate,up,down}_proj.weight
        if name == "model.language_model.embed_tokens.weight":
            return (0, 0, 0, 0)
        m = name.startswith("model.language_model.layers.")
        if m:
            rest = name[len("model.language_model.layers."):]
            try:
                layer_s, rest2 = rest.split(".", 1)
                layer = int(layer_s)
            except ValueError:
                return (9, 0, 0, 0)
            if rest2.startswith("self_attn."):
                sub = rest2[len("self_attn."):]
                order = {"q_proj.weight": 0, "k_proj.weight": 1,
                         "v_proj.weight": 2, "o_proj.weight": 3}
                return (1, layer, 0, order.get(sub, 9))
            if rest2.startswith("mlp."):
                sub = rest2[len("mlp."):]
                order = {"gate_proj.weight": 0, "up_proj.weight": 1,
                         "down_proj.weight": 2}
                return (1, layer, 1, order.get(sub, 9))
            return (1, layer, 2, 0)
        if name == "model.language_model.norm.weight":
            return (2, 0, 0, 0)
        if name == "lm_head.weight":
            return (3, 0, 0, 0)
        if name.startswith("model.t_embedder"):
            return (4, 0, 0, 0)
        if name.startswith("model.x_embedder"):
            return (5, 0, 0, 0)
        if name.startswith("model.final_layer"):
            return (6, 0, 0, 0)
        if name.startswith("model.visual"):
            return (7, 0, 0, 0)
        return (8, 0, 0, 0)

    return sorted(tensors.keys(), key=key)


def main():
    ap = argparse.ArgumentParser(description="safetensors -> GGUF v3 converter")
    ap.add_argument("--source", required=True, help="model dir with safetensors shards")
    ap.add_argument("--output", required=True, help="output .gguf path")
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--variant", default=None,
                    help="provenance variant: dev | dev-2604 | base (default: profile)")
    ap.add_argument("--quantization", default="bf16")
    ap.add_argument("--revision", default="unknown")
    ap.add_argument("--architecture", default="hidream_o1")
    args = ap.parse_args()

    variant = args.variant or args.profile

    tensors = load_shard_index(args.source)
    names = production_order(tensors)
    print(f"[convert] {len(names)} tensors, {sum(t['nbytes'] for t in tensors.values())/1e9:.1f} GB F32")

    w = Writer(args.output)
    w.u32(GGUF_MAGIC)
    w.u32(GGUF_VERSION)
    w.u64(len(names))  # tensor_count

    # metadata KV pairs
    kvs = [
        ("general.architecture", args.architecture),
        ("general.name", f"HiDream-O1-Image-{variant}"),
        ("general.alignment", str(ALIGNMENT)),
        ("hidream.profile", args.profile),
        ("hidream.variant", variant),
        ("hidream.revision", args.revision),
        ("hidream.dtype", "bf16"),
        ("hidream.quantization", args.quantization),
        ("hidream.num_layers", "36"),
        ("hidream.layout_version", "1"),
        ("hidream.source_format", "safetensors"),
    ]
    w.u64(len(kvs))
    for key, value in kvs:
        if key == "general.alignment":
            w.string(key); w.u32(GGUF_TYPE_UINT32); w.u32(ALIGNMENT)
        else:
            w.kv_string(key, value)

    # tensor infos: compute offsets first (relative to tensor_data)
    infos = []
    offset = 0
    for name in names:
        t = tensors[name]
        nbytes = t["nbytes"]
        if t["dtype"] == "F32":
            nbytes = t["nbytes"] // 2  # F32 -> BF16
        offset = align_up(offset, ALIGNMENT)
        infos.append((name, t, offset, nbytes))
        offset += nbytes
    payload_bytes = align_up(offset, ALIGNMENT)

    for name, t, off, nbytes in infos:
        w.string(name)
        w.u32(len(t["shape"]))
        for d in t["shape"]:
            w.u64(d)
        w.u32(GGML_TYPE_BF16)
        w.u64(off)

    # pad to ALIGNMENT before tensor data
    w.pad(ALIGNMENT)

    # tensor data: stream each tensor, casting F32 -> BF16
    sha = hashlib.sha256()
    for name, t, off, nbytes in infos:
        with open(t["path"], "rb") as f:
            f.seek(t["data_begin"])
            raw = f.read(t["nbytes"])
        if t["dtype"] == "F32":
            data = f32_buf_to_bf16_np(raw)
        else:
            data = raw
        w.write(bytes(data))
        sha.update(bytes(data))
        w.pad(ALIGNMENT)
        print(f"  {name} {nbytes/1e6:.1f} MB")

    w.close()
    print(f"[convert] wrote {args.output} ({os.path.getsize(args.output)/1e9:.2f} GB)")
    print(f"[convert] payload sha256: {sha.hexdigest()}")


if __name__ == "__main__":
    main()