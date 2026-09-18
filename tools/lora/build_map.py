#!/usr/bin/env python3
"""Generate the canonical musubi-LoRA -> o1 weight map.

Reads the frozen tensor manifest (config/tensor_manifest_<profile>.json) and
emits config/hidream_lora_map_<profile>.json. The map is the explicit
compatibility boundary between:

  musubi LoRA key prefix   (lora_unet_<module_path with _>)
  upstream state-dict name (model.<module_path>.weight)
  o1 internal role         (block.N.q_proj, ...)
  GGUF tensor name         (blk.NN.attn.q.weight, ...)

Usage:
  python tools/lora/build_map.py \
      --tensor-manifest config/tensor_manifest_full.json \
      --profile full \
      --output config/hidream_lora_map_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    TARGET_LINEAR_SUBMODULES,
    lora_key_for_module,
    upstream_key_for_module,
)

# o1 role / GGUF name derivation for the language-model decoder layers.
ATTN_ROLES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
MLP_ROLES = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def o1_role_and_gguf(module_path: str) -> tuple[str, str]:
    """Map an upstream module path to (o1_role, gguf_name)."""
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$", module_path)
    if m:
        layer = int(m.group(1))
        kind = m.group(2)
        sub = m.group(3)
        if kind == "self_attn" and sub in ATTN_ROLES:
            return f"block.{layer}.{ATTN_ROLES[sub]}_proj", f"blk.{layer:02d}.attn.{ATTN_ROLES[sub]}.weight"
        if kind == "mlp" and sub in MLP_ROLES:
            return f"block.{layer}.{MLP_ROLES[sub]}_proj", f"blk.{layer:02d}.ffn.{MLP_ROLES[sub]}.weight"
    if module_path == "model.language_model.norm":
        return "final_norm", "final.norm.weight"
    if module_path == "model.t_embedder1.mlp.0":
        return "t_embedder.mlp.0", "time.mlp.0.weight"
    if module_path == "model.t_embedder1.mlp.2":
        return "t_embedder.mlp.2", "time.mlp.2.weight"
    if module_path == "model.x_embedder.proj1":
        return "x_embedder.proj1", "pix.in.proj1.weight"
    if module_path == "model.x_embedder.proj2":
        return "x_embedder.proj2", "pix.in.proj2.weight"
    if module_path == "model.final_layer2.linear":
        return "final_layer.linear", "pix.out.linear.weight"
    return "", ""


def build_map(manifest: dict) -> list[dict]:
    tensors = {t["name"]: t for t in manifest["tensors"]}
    entries = []

    # Decoder layers: 36 layers x (4 attn + 3 mlp) linear weights.
    n_layers = 0
    for name in tensors:
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)

    for layer in range(n_layers):
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = f"model.language_model.layers.{layer}.self_attn.{sub}"
            _add_linear(entries, tensors, module)
        for sub in ("gate_proj", "up_proj", "down_proj"):
            module = f"model.language_model.layers.{layer}.mlp.{sub}"
            _add_linear(entries, tensors, module)

    # Pixel / timestep / final layers.
    for module in (
        "model.x_embedder.proj1",
        "model.x_embedder.proj2",
        "model.final_layer2.linear",
        "model.t_embedder1.mlp.0",
        "model.t_embedder1.mlp.2",
    ):
        _add_linear(entries, tensors, module)

    return entries


def _add_linear(entries: list[dict], tensors: dict, module: str) -> None:
    base_tensor = module + ".weight"
    t = tensors.get(base_tensor)
    if t is None:
        return
    role, gguf = o1_role_and_gguf(module)
    entries.append(
        {
            "base_tensor": base_tensor,
            "base_module": module,
            "lora_prefix": lora_key_for_module(module),
            "kind": "linear",
            "shape": t["shape"],
            "o1_role": role,
            "gguf_name": gguf,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="build canonical LoRA map")
    ap.add_argument("--tensor-manifest", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.tensor_manifest) as f:
        manifest = json.load(f)

    entries = build_map(manifest)
    if not entries:
        print(f"error: no linear targets resolved from {args.tensor_manifest}")
        sys.exit(1)

    out = {
        "format": "o1-lora-map-v1",
        "profile": args.profile,
        "source_manifest": os.path.basename(args.tensor_manifest),
        "n_entries": len(entries),
        "entries": entries,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}: {len(entries)} linear targets")


if __name__ == "__main__":
    main()