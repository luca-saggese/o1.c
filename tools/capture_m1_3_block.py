#!/usr/bin/env python3
"""M1.3 decoder block golden fixture capture.

Captures the complete layer-0 text decoder block chain from the frozen Python
oracle (Qwen3VLTextDecoderLayer.forward). This is a single chained V2 capture:
we load the model once and call *module* forward paths only (input_layernorm,
self_attn, post_attention_layernorm, mlp) — never the whole-model transformer
forward. whole_model_forwards remains 0.

The M1.2 GEMM/attn fixtures used raw h_full as input (not chained). M1.3 needs
the true chained block output, so a single fresh oracle load is justified.

Usage:
    .venv/bin/python tools/capture_m1_3_block.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "artifacts" / "m1" / "golden"
ORACLE_SHA = "3237a638a5c2c7be106b0175958f4c0db8c2dfbf"
DEV_REVISION = "b6acc2fe452b3120430620dc4354fa442ee081ea"
PROMPT = "a red fox sits under a cherry blossom tree"
SEED = 42
RES = 64  # FAST_VALIDATION_RES
PATCH_SIZE = 32
TMS_TOKEN_ID = 151673


def run_guard():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "m1_guard.py"), "check-env"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("m1_guard check-env FAILED:\n", r.stdout, r.stderr)
        sys.exit(1)
    print("m1_guard check-env OK")


def bf16_bytes(t):
    t = t.to(torch.bfloat16).cpu().contiguous()
    return t.view(torch.int16).numpy().astype("<u2").tobytes()


def f32_bytes(t):
    return t.float().cpu().contiguous().numpy().tobytes()


def i64_bytes(t):
    return t.long().cpu().contiguous().numpy().tobytes()


class FixtureWriter:
    def __init__(self):
        self.manifest = {
            "schema_version": 1,
            "oracle_sha": ORACLE_SHA,
            "model_profile": "dev",
            "model_revision": DEV_REVISION,
            "capture_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "compute_dtype": "bfloat16",
            "stored_dtype": "float32",
            "seed": SEED,
            "resolution": RES,
            "seq_len": 23,
            "text_seq_len": 19,
            "milestone": "M1.3",
            "fixtures": {},
        }
        self.fixtures = {}

    def add(self, fixture_id, primitive, cls, payloads, params=None):
        bin_path = GOLDEN_DIR / f"{fixture_id}.bin"
        offset = 0
        inputs, outputs = [], []
        for name, dtype, shape, data in payloads:
            entry = {
                "name": name, "file": f"{fixture_id}.bin", "offset": offset,
                "nbytes": len(data), "dtype": dtype, "shape": list(shape),
            }
            offset += len(data)
            if name.startswith("out_"):
                outputs.append(entry)
            else:
                inputs.append(entry)
        bin_path.write_bytes(b"".join(p[3] for p in payloads))
        meta = {
            "fixture_id": fixture_id,
            "primitive": primitive,
            "class": cls,
            "inputs": inputs,
            "outputs": outputs,
            "params": params or {},
        }
        (GOLDEN_DIR / f"{fixture_id}.json").write_text(
            json.dumps(meta, indent=2) + "\n")
        self.fixtures[fixture_id] = meta
        self.manifest["fixtures"][fixture_id] = {
            "file": f"{fixture_id}.json",
            "nbytes": (GOLDEN_DIR / f"{fixture_id}.json").stat().st_size,
            "sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
        }
        print(f"  {fixture_id}: {len(payloads)} tensors, {offset} B")

    def write_manifest(self):
        # Merge M1.2 (existing per-fixture JSONs) entries with this run's
        # fixtures into a single authoritative golden manifest. This must not
        # clobber the M1.2 fixtures registered earlier by Agent's capture.
        merged = {}
        for fjson in GOLDEN_DIR.glob("*.json"):
            if fjson.name == "manifest.json":
                continue
            try:
                meta = json.loads(fjson.read_text())
            except Exception:
                continue
            fid = meta.get("fixture_id")
            if not fid:
                continue
            binpath = GOLDEN_DIR / f"{fid}.bin"
            nbytes = binpath.stat().st_size if binpath.exists() else 0
            merged[fid] = {
                "file": fjson.name,
                "nbytes": nbytes,
                "sha256": (hashlib.sha256(binpath.read_bytes()).hexdigest()
                           if binpath.exists() else ""),
            }
        self.manifest["fixtures"] = merged
        (GOLDEN_DIR / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n")
        print(f"manifest.json: {len(merged)} fixtures (merged M1.2 + M1.3)")


def main():
    run_guard()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FA_VERSION", "0")
    os.environ.setdefault("USE_BF16_ROPE", "0")
    sys.path.insert(0, str(ROOT / "python"))

    import torch
    from transformers import AutoProcessor
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
    from models.pipeline import build_t2i_text_sample, PATCH_SIZE as PS

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    writer = FixtureWriter()

    local_path = str(ROOT / "models" / "dev")
    print("Loading frozen Dev model (single load)...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, _attn_implementation="eager",
    ).eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")

    device = model.device
    dtype = torch.bfloat16
    cfg = model.config
    text_cfg = cfg.text_config
    H = text_cfg.hidden_size            # 4096
    I = text_cfg.intermediate_size      # 12288
    NH = text_cfg.num_attention_heads   # 32
    NKV = text_cfg.num_key_value_heads  # 8
    HD = text_cfg.head_dim              # 128
    S = 23
    TS = 19

    sample = build_t2i_text_sample(
        PROMPT, RES, RES, processor.tokenizer, processor, cfg)
    input_ids = sample["input_ids"].to(device)
    position_ids = sample["position_ids"].to(device)    # [3,1,23]
    token_types = sample["token_types"].to(device)
    vinput_mask = sample["vinput_mask"].to(device)

    # canonical int32/bf16 position ids for the block C ABI
    pos_f32 = position_ids.float()                      # [3,1,23] fp32

    min_val = torch.finfo(torch.bfloat16).min
    causal = torch.full((S, S), min_val, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    gen_positions = token_types[0].bool()
    causal[gen_positions, :] = 0.0
    attn_mask_4d = causal.unsqueeze(0).unsqueeze(0)     # [1,1,23,23] bf16

    # ---- layered block fixtures ----
    lm = model.language_model
    layer0 = lm.layers[0]
    embed = lm.embed_tokens(input_ids)

    # vinput deterministic noise
    torch.manual_seed(SEED + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 1)
    noise = 8.0 * torch.randn(
        (1, 3, RES, RES),
        generator=torch.Generator("cpu").manual_seed(SEED + 1),
    ).to(device, dtype)
    z = torch.nn.functional.pixel_unshuffle(noise, PS)
    z = z.flatten(2).transpose(1, 2)                    # [1,4,3072]
    torch.manual_seed(SEED)

    tms_mask = (input_ids == TMS_TOKEN_ID).unsqueeze(-1).expand_as(embed)
    t = torch.tensor([999], device=device)
    t_emb = model.model.t_embedder1(t)
    h_text = torch.where(tms_mask, t_emb.unsqueeze(1).expand_as(embed), embed)
    x_emb = model.model.x_embedder(z)
    h_full = torch.cat([h_text, x_emb], dim=1)          # [1,23,4096]

    x0 = h_full[0]                                      # [23,4096] block input

    # ---- replicate Qwen3VLTextDecoderLayer.forward exactly ----
    residual = x0
    ln0 = layer0.input_layernorm(x0)                    # input_layernorm(h)
    cos, sin = lm.rotary_emb(h_full, position_ids)      # [1,23,128] bf16
    # decoder layer calls self_attn(ln0[None,...], position_embeddings=(cos,sin),
    #   attention_mask, position_ids, ...); pass seq view.
    attn_out, _ = layer0.self_attn(
        hidden_states=ln0.unsqueeze(0),
        position_embeddings=(cos, sin),
        attention_mask=attn_mask_4d,
        position_ids=position_ids,
        past_key_values=None, use_cache=False,
        cache_position=torch.arange(S, device=device)[None],
    )
    attn_hidden = attn_out[0]                           # [23,4096]
    attn_resid = residual + attn_hidden                 # residual + attn
    post_ln = layer0.post_attention_layernorm(attn_resid)
    mlp_out = layer0.mlp(post_ln)
    block_out = attn_resid + mlp_out                    # final block output

    # capture position ids fp32 and attention mask too (for the block ABI)
    writer.add("block_0", "decoder_block", "D", [
        ("x", "bfloat16", x0.shape, bf16_bytes(x0)),
        ("pos_f32", "float32", pos_f32.shape, f32_bytes(pos_f32)),
        ("mask", "bfloat16", attn_mask_4d.shape, bf16_bytes(attn_mask_4d)),
        ("out_ln0", "bfloat16", ln0.shape, bf16_bytes(ln0)),
        ("out_attn_hidden", "bfloat16", attn_hidden.shape, bf16_bytes(attn_hidden)),
        ("out_attn_resid", "bfloat16", attn_resid.shape, bf16_bytes(attn_resid)),
        ("out_post_ln", "bfloat16", post_ln.shape, bf16_bytes(post_ln)),
        ("out_mlp_out", "bfloat16", mlp_out.shape, bf16_bytes(mlp_out)),
        ("out_block", "bfloat16", block_out.shape, bf16_bytes(block_out)),
    ], {
        "layer": 0, "hidden_size": H, "ff_hidden": I, "heads": NH,
        "kv_heads": NKV, "head_dim": HD, "rms_eps": text_cfg.rms_norm_eps,
        "rope_theta": text_cfg.rope_theta,
        "mrope_section": text_cfg.rope_scaling["mrope_section"],
        "attention_bias": text_cfg.attention_bias,
        "composition": "residual=x; h=input_layernorm(x); "
                       "h=self_attn(h); h=residual+h; residual=h; "
                       "h=post_attention_layernorm(h); h=mlp(h); h=residual+h",
        "chained": True,
        "note": "true chained block output (M1.2 fixtures used raw h_full "
                "as projection input, so this is a fresh single-load capture)",
    })

    writer.write_manifest()

    assert torch.isfinite(block_out).all(), "block output contains NaN/Inf"
    for name, tt in (("ln0", ln0), ("attn_hidden", attn_hidden),
                     ("attn_resid", attn_resid), ("post_ln", post_ln),
                     ("mlp_out", mlp_out)):
        assert torch.isfinite(tt).all(), f"{name} contains NaN/Inf"
    print("sanity checks OK")

    print(f"\nCaptured {len(writer.fixtures)} M1.3 fixture -> {GOLDEN_DIR}")
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402
    sys.exit(main())