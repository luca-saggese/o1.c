"""Shared helpers for the o1.c LoRA toolchain.

The naming and formula contracts here are pinned to musubi-tuner commit
4e7c7149249e7715e9168920feb4c420423abba7 (see musubi.lock):

  - LoRA key prefix:  "lora_unet_" + module_path.replace(".", "_")
  - linear layout:    down [rank, in_dim], up [out_dim, rank]
  - scale:            alpha / rank  (alpha defaults to rank when absent)
  - merge:            W' = W + multiplier * scale * (up @ down)

Never reverse underscores heuristically; always build names from a known
base module path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LORA_PREFIX = "lora_unet_"

# musubi HiDream-O1 T2I target module classes (pinned contract).
T2I_TARGET_CLASSES = [
    "Qwen3VLTextDecoderLayer",
    "BottleneckPatchEmbed",
    "FinalLayer",
    "TimestepEmbedder",
]

# Linear submodules inside each target class, in the order musubi walks them.
# key = class name, value = list of (attribute path, kind)
TARGET_LINEAR_SUBMODULES: Dict[str, List[Tuple[str, str]]] = {
    "Qwen3VLTextDecoderLayer": [
        ("self_attn.q_proj", "linear"),
        ("self_attn.k_proj", "linear"),
        ("self_attn.v_proj", "linear"),
        ("self_attn.o_proj", "linear"),
        ("mlp.gate_proj", "linear"),
        ("mlp.up_proj", "linear"),
        ("mlp.down_proj", "linear"),
    ],
    "BottleneckPatchEmbed": [
        ("proj1", "linear"),
        ("proj2", "linear"),
    ],
    "FinalLayer": [
        ("linear", "linear"),
    ],
    "TimestepEmbedder": [
        ("mlp.0", "linear"),
        ("mlp.2", "linear"),
    ],
}

# Upstream safetensors prefix for the HiDream model.
UPSTREAM_PREFIX = "model."


def lora_key_for_module(module_path: str) -> str:
    """lora_unet_<module_path with dots replaced by underscores>."""
    return LORA_PREFIX + module_path.replace(".", "_")


def upstream_key_for_module(module_path: str) -> str:
    """model.<module_path>.weight"""
    return UPSTREAM_PREFIX + module_path + ".weight"


def module_path_from_lora_key(key: str) -> Optional[str]:
    """Inverse of lora_key_for_module for keys we generated ourselves.

    Only valid for keys produced by this toolchain; never used to guess
    arbitrary third-party names.
    """
    if not key.startswith(LORA_PREFIX):
        return None
    return key[len(LORA_PREFIX):].replace("_", ".")


def parse_lora_spec(spec: str) -> Tuple[str, float]:
    """'path' or 'path:multiplier' -> (path, multiplier)."""
    if ":" in spec:
        path, _, mult = spec.rpartition(":")
        try:
            m = float(mult)
        except ValueError:
            raise ValueError(f"invalid multiplier in {spec!r}")
        if m != m or m in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite multiplier in {spec!r}")
        return path, m
    return spec, 1.0


def sha256_file(path: str, chunk: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass
class LoraEntry:
    """One resolved linear LoRA target."""

    lora_prefix: str          # e.g. lora_unet_model_language_model_layers_0_self_attn_q_proj
    base_module: str          # e.g. model.language_model.layers.0.self_attn.q_proj
    base_tensor: str          # e.g. model.language_model.layers.0.self_attn.q_proj.weight
    kind: str                 # "linear"
    rank: int
    in_dim: int
    out_dim: int
    alpha: float
    multiplier: float = 1.0
    down_key: str = ""
    up_key: str = ""
    alpha_key: str = ""


@dataclass
class LoraAdapter:
    path: str
    entries: List[LoraEntry] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    sha256: str = ""
    base_profile: str = ""

    @property
    def supported_count(self) -> int:
        return len(self.entries)

    @property
    def rank_distribution(self) -> Dict[int, int]:
        d: Dict[int, int] = {}
        for e in self.entries:
            d[e.rank] = d.get(e.rank, 0) + 1
        return d

    @property
    def alpha_distribution(self) -> Dict[float, int]:
        d: Dict[float, int] = {}
        for e in self.entries:
            d[e.alpha] = d.get(e.alpha, 0) + 1
        return d