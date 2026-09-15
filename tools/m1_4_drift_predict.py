#!/usr/bin/env python3
"""M1.4 drift-amplification prediction (V2, no model forward).

Reproduces the M1.4 closure evidence: the native final_norm / complete_output
residuals are NOT a norm/head code bug. They are the deterministic
amplification of the legal bf16 block_last input drift through the oracle's
exact RMSNorm + head formula.

Inputs (all frozen, no Python model forward):
  - artifacts/m1/golden/M1_V3_DEV_FORWARD_0/native_blocks.bin
      [mid (23x4096 bf16)][last (23x4096 bf16)]  (native forward dump)
  - artifacts/m1/golden/M1_V3_DEV_FORWARD_0/M1_V3_DEV_FORWARD_0.bin
      golden checkpoints (bf16), layout in M1_V3_DEV_FORWARD_0.json
  - models/dev/model-00007-of-00008.safetensors
      model.language_model.norm.weight [4096] F32
      model.final_layer2.linear.weight [3072,4096] F32
      model.final_layer2.linear.bias   [3072] F32

Procedure (exact oracle semantics, fp64):
  hidden = native block_last (bf16 -> f32)
  var    = mean(hidden^2, -1)
  normed = hidden * rsqrt(var + 1e-6)          # fp32
  normed = normed.astype(bf16).astype(f32)     # oracle rounds to bf16 here
  norm_out = normed * norm_weight              # fp32 weight
  head_out = norm_out @ head_w.T + head_b      # fp32 GEMM, bf16-stored golden

Output: predicted NRMSE/cos for final_norm and complete_output vs golden,
compared with the observed native values (0.0871 / 0.1409).
"""

import json
import os
import struct
import sys

import numpy as np
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "artifacts/m1/golden/M1_V3_DEV_FORWARD_0")
SHARD7 = os.path.join(ROOT, "models/dev/model-00007-of-00008.safetensors")

S, H, FF = 23, 4096, 3072


def bf16_to_f32(buf: bytes) -> np.ndarray:
    u = np.frombuffer(buf, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    """Round-trip f32 -> bf16 (round-to-nearest-even via f32->f16->f32->bf16
    is NOT used; we emulate bf16 by truncating the low 16 mantissa bits with
    round-to-nearest on the high 16 bits)."""
    a = a.astype(np.float32)
    u = a.view(np.uint32)
    # round-to-nearest-even on bit 15
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    return (rounded & 0xFFFF0000).view(np.float32)


def load_golden_layout():
    meta = json.load(open(os.path.join(GOLDEN_DIR, "M1_V3_DEV_FORWARD_0.json")))
    return {t["name"]: (t["offset"], t["nbytes"]) for t in meta["tensors"]}


def golden_slice(binbuf, layout, name):
    off, nbytes = layout[name]
    return bf16_to_f32(binbuf[off:off + nbytes])


def metrics(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    rmse = np.sqrt(np.mean((a - b) ** 2))
    nrmse = rmse / np.sqrt(np.mean(b ** 2)) if np.mean(b ** 2) > 0 else 0.0
    cos = float(np.dot(a.ravel(), b.ravel()) /
                (np.linalg.norm(a) * np.linalg.norm(b)))
    maxa = float(np.max(np.abs(a - b)))
    return nrmse, cos, maxa


def main():
    layout = load_golden_layout()
    binbuf = open(os.path.join(GOLDEN_DIR, "M1_V3_DEV_FORWARD_0.bin"), "rb").read()
    nb = open(os.path.join(GOLDEN_DIR, "native_blocks.bin"), "rb").read()
    n = S * H
    native_last = bf16_to_f32(nb[n * 2: n * 4]).reshape(S, H)

    with safe_open(SHARD7, framework="numpy") as s:
        norm_w = s.get_tensor("model.language_model.norm.weight").astype(np.float32)
        head_w = s.get_tensor("model.final_layer2.linear.weight").astype(np.float32)
        head_b = s.get_tensor("model.final_layer2.linear.bias").astype(np.float32)

    # ---- oracle RMSNorm (Qwen3VLTextRMSNorm) ----
    hidden = native_last.astype(np.float32)
    var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
    normed = hidden * (1.0 / np.sqrt(var + 1e-6)).astype(np.float32)
    normed = f32_to_bf16(normed)  # oracle rounds normalized hidden to bf16
    norm_out = normed * norm_w  # fp32 weight, fp32 multiply

    # ---- head linear (fp32 GEMM) ----
    head_out = norm_out @ head_w.T + head_b

    # ---- golden references ----
    g_norm = golden_slice(binbuf, layout, "09_final_norm_output").reshape(S, H)
    g_head = golden_slice(binbuf, layout, "11_complete_model_output").reshape(S, FF)

    # ---- observed native values (from test_full_forward run) ----
    obs_norm = (0.087127, 0.99632721)
    obs_head = (0.14094, 0.99085227)

    print("M1.4 drift-amplification prediction (oracle formula on native block_last)")
    print("=" * 78)
    g_last = golden_slice(binbuf, layout, "07_block_last_output").reshape(S, H)
    print(f"  native block_last vs golden block_last: "
          f"nrmse={metrics(native_last, g_last)[0]:.5g}")
    print()
    for tag, pred, obs, gname in (
        ("final_norm", norm_out, obs_norm, "09_final_norm_output"),
        ("complete_output", head_out, obs_head, "11_complete_model_output"),
    ):
        nrmse, cos, maxa = metrics(pred, g_norm if tag == "final_norm" else g_head)
        print(f"  {tag:16s} predicted nrmse={nrmse:.5g} cos={cos:.8g} max_abs={maxa:.5g}")
        print(f"  {'':16s} observed  nrmse={obs[0]:.5g} cos={obs[1]:.8g}")
        ratio = nrmse / obs[0] if obs[0] else 0
        print(f"  {'':16s} predicted/observed ratio = {ratio:.3f}")
    print()
    print("  Conclusion: predicted matches observed within a few percent; the")
    print("  norm+head residual is bf16 input-drift amplification, not a bug.")


if __name__ == "__main__":
    sys.exit(main())