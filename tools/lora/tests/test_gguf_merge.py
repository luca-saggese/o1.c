"""L7: offline GGUF merge test.

Verifies merge_gguf.py on the synthetic fixture:
  - merged GGUF is readable by the C reader (test_gguf)
  - provenance metadata is present
  - merged weights match the Python reference merge bit-exactly (BF16)
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from gguf import read_gguf, read_tensor_payload  # noqa: E402
from safetensors import safe_open  # noqa: E402

BASE_GGUF = os.path.join(ROOT, "artifacts", "models", "hidream-o1-dev-bf16.gguf")
LORA = os.path.join(ROOT, "artifacts", "lora", "synthetic", "synthetic.safetensors")
MAP = os.path.join(ROOT, "config", "hidream_lora_map_dev.json")
OUT = "/tmp/hidream-o1-dev-synthetic-test.gguf"

TARGETS = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.t_embedder1.mlp.0.weight",
    "model.final_layer2.linear.weight",
]


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16)


def read_bf16(gf, name):
    for t in gf.tensors:
        if t.name == name:
            raw = read_tensor_payload(gf, t)
            return np.frombuffer(raw, dtype=np.uint16).copy()
    raise KeyError(name)


def reference_merge():
    """Compute the expected merged BF16 weights in Python."""
    with safe_open(LORA, framework="np") as f:
        keys = list(f.keys())
    out = {}
    for t in TARGETS:
        module = t[: -len(".weight")]
        prefix = "lora_unet_" + module.replace(".", "_")
        with safe_open(LORA, framework="np") as f:
            down = f.get_tensor(prefix + ".lora_down.weight")
            up = f.get_tensor(prefix + ".lora_up.weight")
            alpha = f.get_tensor(prefix + ".alpha")[0]
        rank = down.shape[0]
        delta = (up @ down) * (alpha / rank)
        base = read_bf16(read_gguf(BASE_GGUF), t)
        base_f32 = (base.astype(np.uint32) << 16).view(np.float32)
        merged = base_f32.reshape(delta.shape) + delta
        out[t] = f32_to_bf16(merged)
    return out


def main():
    if not os.path.exists(BASE_GGUF):
        print("SKIP: base GGUF not present")
        return 0
    if not os.path.exists(LORA):
        print("SKIP: synthetic fixture not present")
        return 0

    merge_script = os.path.join(HERE, "..", "merge_gguf.py")
    rc = subprocess.call([
        sys.executable, merge_script,
        "--base", BASE_GGUF,
        "--lora", f"{LORA}:1.0",
        "--output", OUT,
        "--map", MAP,
    ])
    assert rc == 0, "merge_gguf.py failed"

    gf = read_gguf(OUT)
    assert gf.alignment == 256, f"alignment {gf.alignment} != 256"
    meta = gf.metadata
    assert any("lora" in str(k).lower() for k in meta), "no LoRA provenance metadata"

    want = reference_merge()
    for t in TARGETS:
        got = read_bf16(gf, t)
        assert np.array_equal(got, want[t].reshape(-1)), \
            f"{t}: merged weights differ from reference"
    print("test_gguf_merge PASS (3 targets bit-exact, alignment 256, provenance present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())