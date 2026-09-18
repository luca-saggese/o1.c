"""Minimal GGUF v3 reader/writer for the o1.c LoRA toolchain.

Implements only the subset required by the project: magic/version, KV
metadata, tensor descriptors, aligned offsets, BF16 payload copy. No ggml
runtime dependency. Mirrors the C reader in src/io/gguf.c.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

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

TYPE_BYTES = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
}


@dataclass
class GgufTensor:
    name: str
    n_dims: int
    dims: List[int]
    type: int
    offset: int  # relative to tensor_data
    nbytes: int = 0

    def __post_init__(self):
        n = 1
        for d in self.dims:
            n *= d
        self.nbytes = n * TYPE_BYTES.get(self.type, 0)


@dataclass
class GgufFile:
    path: str
    tensor_count: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)
    tensors: List[GgufTensor] = field(default_factory=list)
    alignment: int = 32
    tensor_data_off: int = 0
    payload_bytes: int = 0


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


class _Writer:
    def __init__(self, f):
        self.f = f
        self.pos = 0

    def write(self, data: bytes):
        self.f.write(data)
        self.pos += len(data)

    def u8(self, v): self.write(struct.pack("<B", v))
    def u32(self, v): self.write(struct.pack("<I", v))
    def u64(self, v): self.write(struct.pack("<Q", v))
    def f32(self, v): self.write(struct.pack("<f", v))
    def f64(self, v): self.write(struct.pack("<d", v))

    def string(self, s: str):
        b = s.encode("utf-8")
        self.u64(len(b))
        self.write(b)

    def pad(self, alignment: int):
        pad = (alignment - (self.pos % alignment)) % alignment
        if pad:
            self.write(b"\x00" * pad)
        return pad


def _write_kv(w: _Writer, key: str, value) -> None:
    w.string(key)
    if isinstance(value, str):
        w.u32(GGUF_TYPE_STRING)
        w.string(value)
    elif isinstance(value, bool):
        w.u32(GGUF_TYPE_BOOL)
        w.u8(1 if value else 0)
    elif isinstance(value, int):
        if key == "general.alignment":
            w.u32(GGUF_TYPE_UINT32)
            w.u32(value)
        else:
            w.u32(GGUF_TYPE_UINT64)
            w.u64(value)
    elif isinstance(value, float):
        w.u32(GGUF_TYPE_FLOAT64)
        w.f64(value)
    else:
        raise ValueError(f"unsupported KV value type for {key}: {type(value)}")


def write_gguf(path: str, metadata: Dict[str, object], tensors: List[GgufTensor],
               alignment: int = 256) -> None:
    """Writes a GGUF v3 file. Tensor payloads are read from `tensors[i].data`
    if present, else must be supplied via the `payload` callback."""
    if alignment < 8 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a power of two >= 8, got {alignment}")

    with open(path, "wb") as f:
        w = _Writer(f)
        w.u32(GGUF_MAGIC)
        w.u32(GGUF_VERSION)
        w.u64(len(tensors))
        w.u64(len(metadata))
        for k, v in metadata.items():
            _write_kv(w, k, v)

        # tensor infos
        for t in tensors:
            w.string(t.name)
            w.u32(t.n_dims)
            for d in t.dims:
                w.u64(d)
            w.u32(t.type)
            w.u64(t.offset)

        w.pad(alignment)
        data_start = w.pos

        # payload
        for t in tensors:
            if w.pos - data_start != t.offset:
                raise ValueError(f"tensor {t.name} offset {t.offset} != {w.pos - data_start}")
            data = getattr(t, "data", None)
            if data is None:
                raise ValueError(f"tensor {t.name} has no payload")
            w.write(data)
            w.pad(alignment)


class _Reader:
    def __init__(self, f):
        self.f = f
        self.pos = 0

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("unexpected EOF")
        self.pos += n
        return b

    def u8(self): return struct.unpack("<B", self.read(1))[0]
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def f64(self): return struct.unpack("<d", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8")

    def skip_value(self, vtype: int) -> None:
        if vtype in (0, 1, 7):
            self.read(1)
        elif vtype in (2, 3):
            self.read(2)
        elif vtype in (4, 5, 6):
            self.read(4)
        elif vtype == 8:
            self.string()
        elif vtype == 9:
            atype = self.u32()
            n = self.u64()
            for _ in range(n):
                self.skip_value(atype)
        elif vtype in (10, 11, 12):
            self.read(8)
        else:
            raise ValueError(f"unsupported KV type {vtype}")


def read_gguf(path: str) -> GgufFile:
    gf = GgufFile(path=path)
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"bad GGUF magic {magic:08x}")
        version = r.u32()
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}")
        gf.tensor_count = r.u64()
        n_kv = r.u64()

        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            if vtype == GGUF_TYPE_STRING:
                gf.metadata[key] = r.string()
            elif vtype == GGUF_TYPE_UINT32:
                gf.metadata[key] = r.u32()
            elif vtype == GGUF_TYPE_UINT64:
                gf.metadata[key] = r.u64()
            elif vtype == GGUF_TYPE_BOOL:
                gf.metadata[key] = bool(r.u8())
            elif vtype == GGUF_TYPE_FLOAT64:
                gf.metadata[key] = r.f64()
            else:
                r.skip_value(vtype)

        for _ in range(gf.tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            gf.tensors.append(GgufTensor(name, n_dims, dims, ttype, offset))

        gf.alignment = int(gf.metadata.get("general.alignment", 32))
        gf.tensor_data_off = align_up(r.pos, gf.alignment)
        payload = 0
        for t in gf.tensors:
            payload = max(payload, t.offset + t.nbytes)
        gf.payload_bytes = align_up(payload, gf.alignment)
    return gf


def read_tensor_payload(gf: GgufFile, t: GgufTensor) -> bytes:
    with open(gf.path, "rb") as f:
        f.seek(gf.tensor_data_off + t.offset)
        data = f.read(t.nbytes)
        if len(data) != t.nbytes:
            raise ValueError(f"short read for {t.name}")
        return data