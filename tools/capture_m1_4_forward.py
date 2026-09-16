#!/usr/bin/env python3
"""M1.4 whole-model transformer forward golden capture (V3).

Captures ONE full V3 forward of the frozen HiDream Dev text decoder from the
frozen Python oracle (Qwen3VLForConditionalGeneration._forward_generation).
This is the single planned expensive Python execution of M1.4: it runs the
real `language_model` over all 36 decoder blocks plus final norm and the real
`final_layer2` head, with forward hooks recording the frozen diagnostic
checkpoints in the SAME run:

  01_model_input
  02_embedding_output
  03_target_or_image_embedding_output
  04_timestep_conditioning
  05_block_00_output        (layer 0)
  06_block_mid_output       (layer 17)
  07_block_last_output      (layer 35)
  08_final_norm_input       (= layer 35 output)
  09_final_norm_output      (language_model.norm output)
  10_final_head_input       (= final norm output)
  10b_final_head_output
  11_complete_model_output  (x_pred = final_layer2(norm out))

Everything is bf16 compute. Inputs are explicit/deterministic (frozen prompt,
fixed seed, fixed noise) so no independent random state is needed.

The preprocessing (text embed lookup, t_emb conditioning, x_embedder target
projection, concat) exactly mirrors _forward_generation steps 1-4; the block
loop + final norm use the real `language_model` once, and the head uses the
real `final_layer2` once. Under eval()+no_grad the t2i dummy-vision branch
adds exactly +0.0, so the result is the true no-pixel_values forward.

Usage:
    .venv/bin/python tools/capture_m1_4_forward.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ROOT = ROOT / "artifacts" / "m1" / "golden" / "M1_V3_DEV_FORWARD_0"
COMPARE_ROOT = ROOT / "artifacts" / "m1" / "compare" / "M1_V3_DEV_FORWARD_0"
ORACLE_SHA = "3237a638a5c2c7be106b0175958f4c0db8c2dfbf"
DEV_REVISION = "b6acc2fe452b3120430620dc4354fa442ee081ea"
PROMPT = "a red fox sits under a cherry blossom tree"
SEED = 42
RES = 64            # FAST_VALIDATION_RES
PATCH_SIZE = 32
TMS_TOKEN_ID = 151673
# Timestep domain (frozen oracle semantics, docs/M1_FORWARD_CONTRACT.md §5):
#   scheduler_timestep = 999            (first scheduler step)
#   sigma              = 999/1000 = 0.999
#   model_timestep     = 1 - sigma ≈ 0.001   (what model.forward receives)
#   embedder_input     = model_timestep * 1000 ≈ 1.0
# The embedder multiplies by 1000 internally (TimestepEmbedder.forward).
# NOTE: this fixture is LEGACY/INVALID FOR PIPELINE SEMANTICS — it passes
# scheduler_timestep directly to the embedder (input 999000). Kept for
# history; do NOT use it for Base or any new gate.
SCHEDULER_TIMESTEP = 999.0
MODEL_TIMESTEP = 1.0 - SCHEDULER_TIMESTEP / 1000.0   # ≈ 0.001
MID_LAYER = 17      # of 36 -> "middle" frozen choice
MID_LAYER = 18      # frozen choice: layer 18 of 36 (spec allows 17 or 18)
NOISE_SCALE = 8.0
FIXTURE_ID = "M1_V3_DEV_FORWARD_0"


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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)

    local_path = str(ROOT / "models" / "dev")
    print(f"Loading frozen Dev model (fixture {FIXTURE_ID})...")
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
    NH = text_cfg.num_attention_heads   # 32
    NKV = text_cfg.num_key_value_heads  # 8
    HD = text_cfg.head_dim              # 128
    I = text_cfg.intermediate_size      # 12288
    NLAYERS = text_cfg.num_hidden_layers  # 36
    OUT_DIM = PATCH_SIZE * PATCH_SIZE * 3  # 3072
    assert NLAYERS == 36, f"unexpected layer count {NLAYERS}"

    sample = build_t2i_text_sample(
        PROMPT, RES, RES, processor.tokenizer, processor, cfg)
    input_ids = sample["input_ids"].to(device)              # [1,19]
    position_ids = sample["position_ids"].to(device)        # [3,1,23]
    token_types = sample["token_types"].to(device)          # [1,23]
    vinput_mask = sample["vinput_mask"].to(device)          # [1,23]
    TS = input_ids.shape[-1]                                # 19 (text seq)
    S = position_ids.shape[-1]                              # 23 (total seq)
    image_tokens = S - TS                                    # 4

    # deterministic explicit noise -> vinputs [1, img_tokens, 3072]
    noise = NOISE_SCALE * torch.randn(
        (1, 3, RES, RES),
        generator=torch.Generator("cpu").manual_seed(SEED + 1),
    ).to(device, dtype)
    z = torch.nn.functional.pixel_unshuffle(noise, PS)       # [1,3*4*4*4? no] bf16
    # pixel_unshuffle(noise,32): [1, C*32*32, H/32, W/32] = [1, 3072, 2, 2]
    z = z.flatten(2).transpose(1, 2)                         # [1, 4, 3072]

    # batch-1 causal attention mask (exact replica of _forward_generation)
    min_val = torch.finfo(dtype).min
    causal = torch.full((S, S), min_val, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    gen_positions = token_types[0].bool()
    causal[gen_positions, :] = 0.0
    attention_mask_4d = causal.unsqueeze(0).unsqueeze(0)     # [1,1,23,23]

    lm = model.model.language_model
    # LEGACY semantics: this fixture passes scheduler_timestep directly to the
    # embedder (embedder input = 999000). Kept for history only.
    timestep = torch.tensor([SCHEDULER_TIMESTEP], device=device, dtype=torch.float32)

    # ---- replicate _forward_generation steps 1-4 EXACTLY ----
    # 1. text token embeddings
    embed = lm.embed_tokens(input_ids)                       # [1,19,4096]

    # 3. timestep conditioning (tms_token replacement)
    t_emb = model.model.t_embedder1(timestep)                # [1,4096]
    tms_mask = (input_ids == TMS_TOKEN_ID)                   # [1,19]
    tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(embed)
    t_emb_expanded = t_emb.unsqueeze(1).expand_as(embed)
    h_text = torch.where(tms_mask_3d, t_emb_expanded, embed)  # [1,19,4096]

    # 4. target/image embedding of vinputs
    vinputs_embedded = model.model.x_embedder(z).to(dtype)   # [1,4,4096]

    # concat -> full sequence
    h_full = torch.cat([h_text, vinputs_embedded], dim=1)     # [1,23,4096]

    # ---- forward hooks over the production path ----
    block_outputs = {}

    def make_hook(idx):
        def hook(module, args, out):
            block_outputs[idx] = out[0].detach() if isinstance(out, tuple) \
                else out.detach()
        return hook

    hooks = []
    for idx in (0, MID_LAYER, NLAYERS - 1):
        hooks.append(lm.layers[idx].register_forward_hook(make_hook(idx)))

    # ---- single real V3 forward over all 36 blocks + final norm ----
    norm_input_ref = [None]

    def norm_input_hook(module, args, out):
        norm_input_ref[0] = args[0].detach() if args else None
    hooks.append(lm.norm.register_forward_hook(norm_input_hook))

    t1 = time.time()
    with torch.no_grad():
        out_lm = lm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask_4d,
            inputs_embeds=h_full,
            use_cache=False,
        )
    hidden_states = out_lm.last_hidden_state               # [1,23,4096] norm out
    norm_output = hidden_states.detach()
    t_lm = time.time() - t1

    # final head (real final_layer2)
    t2 = time.time()
    with torch.no_grad():
        x_pred = model.model.final_layer2(hidden_states)     # [1,23,3072]
    t_head = time.time() - t2

    for h in hooks:
        h.remove()

    block0 = block_outputs[0]
    mid = block_outputs[MID_LAYER]
    last = block_outputs[NLAYERS - 1]

    # ---- write golden (single .bin + per-fixture .json + manifest) ----
    payloads = [
        ("01_model_input", "int64", input_ids.shape, i64_bytes(input_ids), "in"),
        ("02_embedding_output", "bfloat16", embed.shape, bf16_bytes(embed), "ckpt"),
        ("03_target_or_image_embedding_output", "bfloat16",
         vinputs_embedded.shape, bf16_bytes(vinputs_embedded), "ckpt"),
        ("04_timestep_conditioning", "bfloat16", h_text.shape,
         bf16_bytes(h_text), "ckpt"),
        ("05_block_00_output", "bfloat16", block0.shape, bf16_bytes(block0), "ckpt"),
        ("06_block_mid_output", "bfloat16", mid.shape, bf16_bytes(mid), "ckpt"),
        ("07_block_last_output", "bfloat16", last.shape, bf16_bytes(last), "ckpt"),
        ("08_final_norm_input", "bfloat16", last.shape, bf16_bytes(last), "ckpt"),
        ("09_final_norm_output", "bfloat16", norm_output.shape,
         bf16_bytes(norm_output), "ckpt"),
        ("10_final_head_input", "bfloat16", norm_output.shape,
         bf16_bytes(norm_output), "ckpt"),
        ("10b_final_head_output", "bfloat16", x_pred.shape,
         bf16_bytes(x_pred), "ckpt"),
        ("11_complete_model_output", "bfloat16", x_pred.shape,
         bf16_bytes(x_pred), "out"),
    ]

    bin_path = GOLDEN_ROOT / f"{FIXTURE_ID}.bin"
    offset = 0
    rows = []
    for name, dt, shape, data, role in payloads:
        rows.append({"name": name, "dtype": dt, "shape": list(shape),
                     "offset": offset, "nbytes": len(data), "role": role})
        offset += len(data)
    bin_path.write_bytes(b"".join(p[3] for p in payloads))
    bin_sha = sha256_file(bin_path)

    meta = {
        "fixture_id": FIXTURE_ID,
        "schema_version": 1,
        "milestone": "M1.4",
        "validation_level": "V3",
        "oracle_sha": ORACLE_SHA,
        "model_profile": "dev",
        "model_revision": DEV_REVISION,
        "resolution": [RES, RES],
        "batch": 1,
        "compute_dtype": "bfloat16",
        "stored_dtype": "float32_bf16pairs",
        "seed": SEED,
        "prompt": PROMPT,
        "scheduler_timestep": SCHEDULER_TIMESTEP,
        "sigma": SCHEDULER_TIMESTEP / 1000.0,
        "model_timestep": MODEL_TIMESTEP,
        "timestep_embedder_input": MODEL_TIMESTEP * 1000.0,
        "timestep_semantics": "LEGACY/INVALID FOR PIPELINE SEMANTICS: "
                              "scheduler_timestep passed directly to embedder "
                              "(input 999000). Correct pipeline passes "
                              "model_timestep=1-sigma≈0.001; embedder x1000.",
        "noise_scale": NOISE_SCALE,
        "text_seq_len": TS,
        "image_tokens": image_tokens,
        "seq_len": S,
        "mid_layer": MID_LAYER,
        "num_layers": NLAYERS,
        "hidden": H, "heads": NH, "kv_heads": NKV, "head_dim": HD,
        "ff_hidden": I, "out_dim": OUT_DIM,
        "mrope_section": text_cfg.rope_scaling["mrope_section"],
        "mrope_interleaved": text_cfg.rope_scaling.get("mrope_interleaved", True),
        "rope_theta": text_cfg.rope_theta,
        "rms_eps": text_cfg.rms_norm_eps,
        "window": "whole-model bf16 forward (equal to _forward_generation)",
        "tensors": rows,
        "bin": {"file": f"{FIXTURE_ID}.bin", "nbytes": offset, "sha256": bin_sha},
        "native_forward_wall_s": round(float(t1 - t0), 3),
        "lm_forward_wall_s": round(t_lm, 3),
        "head_wall_s": round(t_head, 3),
        "capture_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (GOLDEN_ROOT / f"{FIXTURE_ID}.json").write_text(
        json.dumps(meta, indent=2) + "\n")

    # combined golden manifest for the fixture
    manifest = {"fixture": FIXTURE_ID, "files": {
        f"{FIXTURE_ID}.json": sha256_file(GOLDEN_ROOT / f"{FIXTURE_ID}.json"),
        f"{FIXTURE_ID}.bin": bin_sha,
    }, "tensor_count": len(rows)}
    (GOLDEN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    # ---- sanity ----
    assert torch.isfinite(x_pred).all(), "complete output NaN/Inf"
    for name, tt in (
        ("embed", embed), ("t_emb", t_emb), ("vinputs_embedded", vinputs_embedded),
        ("h_text", h_text), ("h_full", h_full), ("block0", block0),
        ("mid", mid), ("last", last), ("norm_output", norm_output),
        ("x_pred", x_pred)):
        assert torch.isfinite(tt).all(), f"{name} contains NaN/Inf"

    print("\n=== checkpoint plan (frozen) ===")
    for name, _dt, _shape, _data, _role in payloads:
        print(f"  {name:34s} shape={list(_shape)} {len(_data)}B")
    print(f"\nseq_len={S} text={TS} img={image_tokens} scheduler_timestep={SCHEDULER_TIMESTEP}")
    print(f"lm forward {t_lm:.3f}s, head {t_head:.3f}s, total {time.time()-t0:.2f}s")
    print(f"golden -> {GOLDEN_ROOT}  [{offset} B, sha {bin_sha[:16]}...]")
    print("sanity checks OK (all finite)")
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402
    sys.exit(main())