#!/usr/bin/env python3
"""
Generate the synthetic LoRA fixture used by the test ladder.

Three linear targets (T2I, Full, BF16):
  - language_model.layers.0.self_attn.q_proj   [4096,4096] rank 4
  - t_embedder1.mlp.0                          [4096,256]  rank 4, alpha 8
  - final_layer2.linear                        [3072,4096] rank 4

Values are deterministic (seeded) so the fixture is reproducible.
"""

import argparse
import json
import os

import numpy as np
from safetensors.numpy import save_file


def make_target(prefix: str, out_dim: int, in_dim: int, rank: int,
                alpha: float, rng: np.random.Generator):
    down = rng.standard_normal((rank, in_dim)).astype(np.float32)
    up = rng.standard_normal((out_dim, rank)).astype(np.float32)
    return {
        f"{prefix}.alpha": np.asarray(alpha, dtype=np.float32),
        f"{prefix}.lora_down.weight": down,
        f"{prefix}.lora_up.weight": up,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="artifacts/lora/synthetic/synthetic.safetensors")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tensors = {}
    tensors.update(make_target(
        "lora_unet_model_language_model_layers_0_self_attn_q_proj",
        4096, 4096, 4, 4.0, rng))
    tensors.update(make_target(
        "lora_unet_model_t_embedder1_mlp_0",
        4096, 256, 4, 8.0, rng))
    tensors.update(make_target(
        "lora_unet_model_final_layer2_linear",
        3072, 4096, 4, 4.0, rng))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_file(tensors, args.output)
    print(f"wrote {args.output}")

    # sanity: re-read and verify
    from safetensors import safe_open
    with safe_open(args.output, framework="np") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            assert t.size > 0, f"{k}: empty"
    print("sanity: safe_open OK")


if __name__ == "__main__":
    main()