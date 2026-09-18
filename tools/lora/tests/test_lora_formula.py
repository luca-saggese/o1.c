"""L0: LoRA merge formula test.

Verifies the pinned musubi contract:

    W' = W + multiplier * (alpha/rank) * (up @ down)

with a tiny synthetic case (N=3, K=5, R=2) and known values.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import lora_key_for_module, module_path_from_lora_key  # noqa: E402


def merge_linear(w, up, down, alpha, rank, multiplier=1.0):
    scale = alpha / rank
    return w + multiplier * scale * (up @ down)


def test_formula():
    rng = np.random.default_rng(0)
    N, K, R = 3, 5, 2
    w = rng.standard_normal((N, K))
    down = rng.standard_normal((R, K))
    up = rng.standard_normal((N, R))
    alpha = 4.0
    multiplier = 0.8

    got = merge_linear(w, up, down, alpha, R, multiplier)
    want = w + multiplier * (alpha / R) * (up @ down)
    assert np.allclose(got, want), "formula mismatch"
    # alpha == rank -> scale 1
    got2 = merge_linear(w, up, down, R, R, 1.0)
    assert np.allclose(got2, w + up @ down), "alpha==rank scale should be 1"
    print("test_formula PASS")


def test_naming():
    assert lora_key_for_module("model.language_model.layers.0.self_attn.q_proj") == (
        "lora_unet_model_language_model_layers_0_self_attn_q_proj"
    )
    assert lora_key_for_module("model.t_embedder1.mlp.0") == "lora_unet_model_t_embedder1_mlp_0"
    assert lora_key_for_module("model.final_layer2.linear") == "lora_unet_model_final_layer2_linear"
    # reverse underscore guessing is NOT supported (not reversible)
    assert module_path_from_lora_key("lora_unet_model_final_layer2_linear") != "model.final_layer2.linear"
    print("test_naming PASS")


if __name__ == "__main__":
    test_formula()
    test_naming()
    print("ALL PASS")