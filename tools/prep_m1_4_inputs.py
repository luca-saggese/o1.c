#!/usr/bin/env python3
"""M1.4 native-forward INPUT preparation (no whole-model forward).

The frozen V3 golden (M1_V3_DEV_FORWARD_0) already holds the 11 diagnostic
OUTPUT checkpoints captured by capture_m1_4_forward.py. The native full
forward additionally needs the explicit deterministic INPUT tensors the oracle
reconstructed (position_ids fp32, causal mask, vinputs, timestep). These are
pure function of the frozen prompt/seed (build_t2i_text_sample + fixed-noise
pixel_unshuffle); reconstructing them does NOT run a transformer forward and
therefore does not consume the M1.4 V3 run budget.

This tool writes those inputs into the golden dir and cross-checks the
reconstructed text embedding (embed_tokens lookup only, no decoder) against
the frozen 02_embedding_output checkpoint to prove the inputs are consistent
with the golden without a whole-model run.

Usage:
    .venv/bin/python tools/prep_m1_4_inputs.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "artifacts" / "m1" / "golden" / "M1_V3_DEV_FORWARD_0"
ORACLE_SHA = "3237a638a5c2c7be106b0175958f4c0db8c2dfbf"
PATCH_SIZE = 32
SEED = 42
RES = 64
NOISE_SCALE = 8.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FA_VERSION", "0")
    sys.path.insert(0, str(ROOT / "python"))

    import torch
    from transformers import AutoProcessor
    from models.pipeline import build_t2i_text_sample
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    local_path = str(ROOT / "models" / "dev")

    # build the canonical sample (tokenizer only; no decoder forward)
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, _attn_implementation="eager").eval()
    cfg = model.config
    sample = build_t2i_text_sample(
        "a red fox sits under a cherry blossom tree", RES, RES,
        processor.tokenizer, processor, cfg)

    input_ids = sample["input_ids"]                # [1,19] int64
    position_ids = sample["position_ids"]          # [3,1,23] int64
    token_types = sample["token_types"]            # [1,23]
    vinput_mask = sample["vinput_mask"]            # [1,23]
    TS = input_ids.shape[-1]
    S = position_ids.shape[-1]
    image_tokens = S - TS

    # causal attention mask (exact replica of _forward_generation)
    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    causal = torch.full((S, S), min_val, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal[token_types[0].bool(), :] = 0.0
    mask = causal.unsqueeze(0).unsqueeze(0)        # [1,1,23,23] bf16

    # vinputs: deterministic fixed noise -> pixel_unshuffle
    noise = NOISE_SCALE * torch.randn(
        (1, 3, RES, RES),
        generator=torch.Generator("cpu").manual_seed(SEED + 1)).to(torch.bfloat16)
    z = torch.nn.functional.pixel_unshuffle(noise, PATCH_SIZE).flatten(2).transpose(1, 2)
    vinputs = z.contiguous()                       # [1,4,3072] bf16

    timestep = torch.tensor([999.0], dtype=torch.float32)  # [1] f32
    pos_f32 = position_ids.float().contiguous()    # [3,1,23] f32
    def bf16_b(t):
        return t.to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().astype("<u2").tobytes()
    def f32_b(t):
        return t.float().cpu().contiguous().numpy().tobytes()
    def i64_b(t):
        return t.long().cpu().contiguous().numpy().tobytes()

    payloads = [
        ("pos_f32", "float32", list(pos_f32.shape), f32_b(pos_f32)),
        ("mask", "bfloat16", list(mask.shape), bf16_b(mask)),
        ("vinputs", "bfloat16", list(vinputs.shape), bf16_b(vinputs)),
        ("timestep", "float32", list(timestep.shape), f32_b(timestep)),
        ("token_types", "int64", list(token_types.shape), i64_b(token_types)),
        ("vinput_mask", "int64", list(vinput_mask.shape), i64_b(vinput_mask)),
    ]
    bin_path = GOLDEN / "inputs.bin"
    offset = 0
    rows = []
    for name, dt, shape, data in payloads:
        rows.append({"name": name, "dtype": dt, "shape": shape,
                     "offset": offset, "nbytes": len(data)})
        offset += len(data)
    bin_path.write_bytes(b"".join(p[3] for p in payloads))
    # LEGACY fixture: the "timestep" input tensor holds scheduler_timestep
    # (999.0), which the legacy native test feeds to hd_forward directly.
    # Correct pipeline semantics: model_timestep = 1 - 999/1000 ≈ 0.001.
    interim = {"oracle_sha": ORACLE_SHA, "fixture_id": "M1_V3_DEV_FORWARD_0",
               "seed": SEED, "scheduler_timestep": 999.0,
               "sigma": 0.999, "model_timestep": 0.001,
               "timestep_embedder_input": 1.0,
               "timestep_semantics": "LEGACY/INVALID FOR PIPELINE SEMANTICS: "
                                     "input tensor holds scheduler_timestep "
                                     "(999.0), not model_timestep",
               "ttime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "note": "reconstructed inputs; no transformer forward run",
               "tensors": rows,
               "bin": {"file": "inputs.bin", "nbytes": offset,
                       "sha256": sha256_file(bin_path)}}
    (GOLDEN / "inputs.json").write_text(json.dumps(interim, indent=2) + "\n")

    # ---- cross-check: text embedding must match frozen 02 checkpoint ----
    with torch.no_grad():
        embed = model.model.language_model.embed_tokens(input_ids.cuda())
    embed_bf16 = embed.cpu().to(torch.bfloat16).contiguous()

    # read frozen 02_embedding_output
    with open(GOLDEN / "M1_V3_DEV_FORWARD_0.json") as f:
        meta = json.load(f)
    row02 = next(t for t in meta["tensors"] if t["name"] == "02_embedding_output")
    with open(GOLDEN / "M1_V3_DEV_FORWARD_0.bin", "rb") as f:
        f.seek(row02["offset"])
        ref_bytes = f.read(row02["nbytes"])
    ref = torch.frombuffer(ref_bytes, dtype=torch.int16).view(torch.bfloat16)
    ref = ref.reshape(embed_bf16.shape)   # [1,19,4096]
    d = (embed_bf16.float() - ref.float()).abs()
    print(f"embed cross-check: max_abs={d.max().item():.6g} "
          f"(expect ~0 for bf16 embed of identical ids)")
    assert d.max().item() < 1.0, "reconstructed input mismatch with golden"
    print("inputs written ->", GOLDEN / "inputs.bin")
    for r in rows:
        print(f"  {r['name']:14s} {r['shape']} {r['nbytes']}B")
    return 0


if __name__ == "__main__":
    sys.exit(main())