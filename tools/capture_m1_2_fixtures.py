#!/usr/bin/env python3
"""M1.2 golden fixture capture.

Captures deterministic V2 golden tensors from the frozen Python oracle for the
reference CUDA transformer primitives. Follows docs/M1_2_GOLDEN_CONTRACT.md.

No transformer forward, no denoising, no image generation: every fixture is
built by calling individual module internals (projections, norms, rotary,
attention interface) directly.

Usage:
    .venv/bin/python tools/capture_m1_2_fixtures.py
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
TIMESTEP_TOKEN_NUM = 1
TMS_TOKEN_ID = 151673


def run_guard():
    """Run tools/m1_guard.py check-env; abort on failure."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "m1_guard.py"), "check-env"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("m1_guard check-env FAILED:\n", r.stdout, r.stderr)
        sys.exit(1)
    print("m1_guard check-env OK")


def bf16_bytes(t):
    """Raw little-endian bf16 bytes for a CUDA tensor."""
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
            "fixtures": {},
        }
        self.fixtures = {}

    def add(self, fixture_id, primitive, cls, payloads, params=None):
        """payloads: list of (name, dtype, shape, bytes)."""
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
        (GOLDEN_DIR / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n")
        print(f"manifest.json: {len(self.fixtures)} fixtures")


def main():
    run_guard()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FA_VERSION", "0")
    os.environ.setdefault("USE_BF16_ROPE", "0")
    sys.path.insert(0, str(ROOT / "python"))

    import torch
    from transformers import AutoProcessor
    from models.qwen3_vl_transformers import (
        Qwen3VLForConditionalGeneration, eager_attention_forward,
        apply_rotary_pos_emb, rotate_half,
    )
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
    H = text_cfg.hidden_size          # 4096
    I = text_cfg.intermediate_size    # 12288
    NH = text_cfg.num_attention_heads  # 32
    NKV = text_cfg.num_key_value_heads  # 8
    HD = text_cfg.head_dim            # 128
    GROUPS = NH // NKV                # 4
    EPS = text_cfg.rms_norm_eps       # 1e-6
    S = 23                            # seq len
    TS = 19                           # text seq len

    # ---- canonical sample (t2i) ----
    sample = build_t2i_text_sample(
        PROMPT, RES, RES, processor.tokenizer, processor, cfg)
    input_ids = sample["input_ids"].to(device)          # [1,19]
    position_ids = sample["position_ids"].to(device)    # [3,1,23]
    token_types = sample["token_types"].to(device)      # [1,23]
    vinput_mask = sample["vinput_mask"].to(device)      # [1,23] bool

    # ---- Class A: layout fixtures ----
    writer.add("pos_ids_t2i", "pos_ids", "A", [
        ("pos_ids", "int64", position_ids.shape, i64_bytes(position_ids)),
    ], {"source": "get_rope_index_fix_point"})
    writer.add("token_types", "token_types", "A", [
        ("token_types", "int64", token_types.shape, i64_bytes(token_types)),
    ], {"tms_pos": 18, "gen_pos": [19, 20, 21, 22]})
    writer.add("vinput_mask", "vinput_mask", "A", [
        ("vinput_mask", "int64", vinput_mask.shape, i64_bytes(vinput_mask.long())),
    ], {"gen_positions": [19, 20, 21, 22]})

    # eager 4D mask: 0.0 where kv<=q else finfo(bf16).min
    min_val = torch.finfo(torch.bfloat16).min
    causal = torch.full((S, S), min_val, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    gen_positions = token_types[0].bool()
    causal[gen_positions, :] = 0.0
    attn_mask_4d = causal.unsqueeze(0).unsqueeze(0)  # [1,1,23,23]
    writer.add("attn_mask_4d", "attn_mask", "A", [
        ("mask", "bfloat16", attn_mask_4d.shape, bf16_bytes(attn_mask_4d)),
    ], {"pattern": "0.0 where kv<=q or q in gen_positions else finfo(bf16).min"})

    mrope_section = text_cfg.rope_scaling["mrope_section"]
    writer.add("mrope_sections", "mrope_section", "A", [
        ("sections", "int64", [3], i64_bytes(torch.tensor(mrope_section))),
    ], {"interleaved": True, "rope_theta": text_cfg.rope_theta})

    gqa_map = torch.arange(NKV, device=device).repeat_interleave(GROUPS)
    writer.add("gqa_map", "gqa_head_map", "A", [
        ("head_to_kv", "int64", [NH], i64_bytes(gqa_map)),
    ], {"groups": GROUPS, "n_heads": NH, "n_kv": NKV})

    # ---- build hidden states for layer 0 ----
    lm = model.language_model
    layer0 = lm.layers[0]
    embed = lm.embed_tokens(input_ids)  # [1,19,4096] bf16

    # vinputs: deterministic noise z (seed+1, matching pipeline) patchified
    torch.manual_seed(SEED + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 1)
    noise = 8.0 * torch.randn(
        (1, 3, RES, RES), generator=torch.Generator("cpu").manual_seed(SEED + 1),
    ).to(device, dtype)
    z = torch.nn.functional.pixel_unshuffle(noise, PS)  # [1,3*PS*PS, H/PS, W/PS]
    z = z.flatten(2).transpose(1, 2)                    # [1,4,3072]
    torch.manual_seed(SEED)

    # ---- Class A: patchify roundtrip ----
    writer.add("patchify_roundtrip", "patchify", "A", [
        ("z", "bfloat16", z.shape, bf16_bytes(z)),
        ("out_patches", "bfloat16", z.shape, bf16_bytes(z)),
    ], {"patch_size": PS, "in_channels": 3, "note": "identity roundtrip at 64x64"})

    # ---- Class A: head split/merge roundtrip ----
    # Contract 5.1: q [1,23,32,128] -> split [32,23,128] -> merged back
    q_rand = torch.randn(1, S, NH, HD, device=device, dtype=dtype)  # [1,23,32,128]
    q_split = q_rand.transpose(1, 2)  # [1,32,23,128] (keep batch dim for contiguity)
    q_flat = q_split.contiguous().view(NH, S, HD)  # [32,23,128] seq-major per head
    q_merged = q_flat.transpose(0, 1).reshape(1, S, -1)  # [1,23,4096]
    writer.add("head_split_merge", "head_split", "A", [
        ("q", "bfloat16", q_rand.shape, bf16_bytes(q_rand)),
        ("out_q_flat", "bfloat16", q_flat.shape, bf16_bytes(q_flat)),
        ("out_merged", "bfloat16", q_merged.shape, bf16_bytes(q_merged)),
    ], {"n_heads": NH, "head_dim": HD, "layout": "seq-major [H,S,D] per contract 5.1"})

    # ---- timestep embedding ----
    t = torch.tensor([999], device=device)  # first scheduler timestep
    t_emb = model.model.t_embedder1(t)            # [1,4096] bf16 (full mlp: silu included)
    writer.add("t_emb_0", "timestep_embed", "B", [
        ("t", "int64", [1], i64_bytes(t)),
        ("out_t_emb", "bfloat16", t_emb.shape, bf16_bytes(t_emb)),
    ], {"timestep": 999, "freq_size": 256, "max_period": 10000})

    # ---- gemm_t_emb_0: t_embedder1.mlp[0] Linear(256,4096) on the freq embedding ----
    # oracle: t_freq = self.timestep_embedding(t*1000, 256); t_emb = self.mlp[0](t_freq)
    t_freq = model.model.t_embedder1.timestep_embedding(t * 1000, 256)  # [1,256] fp32
    t_emb_lin = model.model.t_embedder1.mlp[0](t_freq.to(model.model.t_embedder1.mlp[0].weight.dtype))  # [1,4096]
    writer.add("gemm_t_emb_0", "linear", "C", [
        ("x", "float32", [1, 256], f32_bytes(t_freq)),
        ("out_y", "bfloat16", t_emb_lin.shape, bf16_bytes(t_emb_lin)),
    ], {"weight": "t_embedder1.mlp.0.weight", "weight_stored_shape": [4096, 256],
        "transpose_w": True, "logical_input": [1, 256], "output_shape": [1, 4096],
        "compute_dtype": "bfloat16", "accum_dtype": "float32",
        "note": "pre-SiLU linear_1; freq input fp32 cast to w.dtype (bf16)"})

    # ---- x_embedder (patch embed) ----
    x_emb = model.model.x_embedder(z)  # [1,4,4096] bf16
    writer.add("x_embedder", "patch_embed", "B", [
        ("z", "bfloat16", z.shape, bf16_bytes(z)),
        ("out_x_emb", "bfloat16", x_emb.shape, bf16_bytes(x_emb)),
    ], {"proj1": "model.model.x_embedder.proj1.weight", "proj2": "model.model.x_embedder.proj2.weight"})

    # ---- final projection ----
    # NOTE contract 5.2 lists x [1,23,4096] -> [1,23,4096], but the oracle
    # FinalLayer.linear is Linear(4096, out_channels*PS^2) = (4096, 3072).
    # Captured oracle truth: [1,23,4096] -> [1,23,3072]. Documented deviation.
    x_final_in = torch.randn(1, S, H, device=device, dtype=dtype)
    x_pred = model.model.final_layer2(x_final_in)  # [1,23,3072]
    writer.add("final_proj_0", "final_proj", "C", [
        ("x", "bfloat16", x_final_in.shape, bf16_bytes(x_final_in)),
        ("out_x_pred", "bfloat16", x_pred.shape, bf16_bytes(x_pred)),
    ], {"weight": "model.model.final_layer2.linear.weight", "bias": "model.model.final_layer2.linear.bias",
        "weight_stored_shape": [3072, 4096], "transpose_w": True,
        "output_shape": [S, 3072], "compute_dtype": "bfloat16", "accum_dtype": "float32",
        "deviation": "contract said [1,23,4096]->[1,23,4096]; oracle is [1,23,4096]->[1,23,3072] (out=3*32*32)"})

    # ---- layer 0 hidden states (text only, then full seq) ----
    # text hidden states: embed + t_emb at tms position
    tms_mask = (input_ids == TMS_TOKEN_ID)  # [1,19]
    tms_mask_3d = tms_mask.unsqueeze(-1).expand_as(embed)
    t_emb_expanded = t_emb.unsqueeze(1).expand_as(embed)
    h_text = torch.where(tms_mask_3d, t_emb_expanded, embed)  # [1,19,4096]

    # full seq: text + vinputs embedded -> this is the model's real input to decoder
    h_full = torch.cat([h_text, x_emb], dim=1)  # [1,23,4096]

    # Contract 5.2 x_embed_0: z -> full decoder input sequence [1,23,4096] (incl t_emb add)
    writer.add("x_embed_0", "patch_embed", "B", [
        ("z", "bfloat16", z.shape, bf16_bytes(z)),
        ("out_h_full", "bfloat16", h_full.shape, bf16_bytes(h_full)),
    ], {"note": "z -> x_embedder -> append to text (with t_emb at tms) -> [1,23,4096]",
        "seq_len": S, "text_seq_len": TS})

    # ---- RMSNorm fixtures ----
    # Contract 5.2: rmsnorm x [1,23,4096] -> y
    x_rms = h_full  # [1,23,4096]
    y_rms0 = layer0.input_layernorm(x_rms)
    writer.add("rmsnorm_0", "rmsnorm", "B", [
        ("x", "bfloat16", x_rms.shape, bf16_bytes(x_rms)),
        ("out_y", "bfloat16", y_rms0.shape, bf16_bytes(y_rms0)),
    ], {"eps": EPS, "weight": "model.language_model.layers.0.input_layernorm.weight"})

    y_rms1 = layer0.post_attention_layernorm(x_rms)
    writer.add("rmsnorm_1", "rmsnorm", "B", [
        ("x", "bfloat16", x_rms.shape, bf16_bytes(x_rms)),
        ("out_y", "bfloat16", y_rms1.shape, bf16_bytes(y_rms1)),
    ], {"eps": EPS, "weight": "model.language_model.layers.0.post_attention_layernorm.weight"})

    # ---- Q/K norm fixtures (head_dim 128) ----
    q_flat = torch.randn(S, NH * HD, device=device, dtype=dtype)
    q_view = q_flat.view(S, NH, HD)
    q_normed = layer0.self_attn.q_norm(q_view)
    writer.add("qnorm_0", "rmsnorm", "B", [
        ("x", "bfloat16", q_view.shape, bf16_bytes(q_view)),
        ("out_y", "bfloat16", q_normed.shape, bf16_bytes(q_normed)),
    ], {"eps": EPS, "weight": "model.language_model.layers.0.self_attn.q_norm.weight",
        "head_dim": HD})

    k_flat = torch.randn(S, NKV * HD, device=device, dtype=dtype)
    k_view = k_flat.view(S, NKV, HD)
    k_normed = layer0.self_attn.k_norm(k_view)
    writer.add("k_norm_0", "rmsnorm", "B", [
        ("x", "bfloat16", k_view.shape, bf16_bytes(k_view)),
        ("out_y", "bfloat16", k_normed.shape, bf16_bytes(k_normed)),
    ], {"eps": EPS, "weight": "model.language_model.layers.0.self_attn.k_norm.weight",
        "head_dim": HD})

    # ---- SiLU / SwiGLU ----
    x_silu = torch.randn(1, TS, I, device=device, dtype=dtype)
    y_silu = torch.nn.functional.silu(x_silu)
    writer.add("silu_0", "silu", "B", [
        ("x", "bfloat16", x_silu.shape, bf16_bytes(x_silu)),
        ("out_y", "bfloat16", y_silu.shape, bf16_bytes(y_silu)),
    ], {})

    gate = torch.randn(1, TS, I, device=device, dtype=dtype)
    up = torch.randn(1, TS, I, device=device, dtype=dtype)
    y_swiglu = torch.nn.functional.silu(gate) * up
    writer.add("swiglu_0", "swiglu", "B", [
        ("gate", "bfloat16", gate.shape, bf16_bytes(gate)),
        ("up", "bfloat16", up.shape, bf16_bytes(up)),
        ("out_y", "bfloat16", y_swiglu.shape, bf16_bytes(y_swiglu)),
    ], {})

    # ---- GEMM fixtures (layer 0, real weights) ----
    x_gemm = h_full[0]  # [23,4096] bf16
    attn0 = layer0.self_attn

    def gemm(name, proj, wname, out_dim, wshape, tw, x_in=None):
        if x_in is None:
            x_in = x_gemm  # [23,4096]
        y = proj(x_in)  # [23,out_dim]
        writer.add(name, "linear", "C", [
            ("x", "bfloat16", x_in.shape, bf16_bytes(x_in)),
            ("out_y", "bfloat16", y.shape, bf16_bytes(y)),
        ], {"weight": wname, "weight_stored_shape": wshape, "transpose_w": tw,
            "logical_input": list(x_in.shape), "output_shape": [S, out_dim],
            "compute_dtype": "bfloat16", "accum_dtype": "float32"})

    gemm("gemm_q_proj_0", attn0.q_proj, "model.language_model.layers.0.self_attn.q_proj.weight",
         NH * HD, [NH * HD, H], True)
    gemm("gemm_k_proj_0", attn0.k_proj, "model.language_model.layers.0.self_attn.k_proj.weight",
         NKV * HD, [NKV * HD, H], True)
    gemm("gemm_v_proj_0", attn0.v_proj, "model.language_model.layers.0.self_attn.v_proj.weight",
         NKV * HD, [NKV * HD, H], True)
    gemm("gemm_o_proj_0", attn0.o_proj, "model.language_model.layers.0.self_attn.o_proj.weight",
         H, [H, H], True)
    gemm("gemm_gate_0", layer0.mlp.gate_proj, "model.language_model.layers.0.mlp.gate_proj.weight",
         I, [I, H], True)
    gemm("gemm_up_0", layer0.mlp.up_proj, "model.language_model.layers.0.mlp.up_proj.weight",
         I, [I, H], True)
    x_down = torch.randn(1, S, I, device=device, dtype=dtype)  # [1,23,12288] (I in-features for down)
    gemm("gemm_down_0", layer0.mlp.down_proj, "model.language_model.layers.0.mlp.down_proj.weight",
         H, [H, I], True, x_in=x_down[0])

    # oracle rotary.forward(x, position_ids) returns cos/sin of shape [bs, seq, head_dim]
    # = [1,23,128], cast to x.dtype (bf16 here). Contract 5.2 said [1,32,23,128] fp32;
    # oracle truth is [1,23,128] bf16. Documented deviation.
    rotary = lm.rotary_emb
    cos, sin = rotary(h_full, position_ids)  # [1,23,128] bf16 (cast to x.dtype)
    writer.add("rope_cos_sin", "mrope", "B", [
        ("pos_ids", "int64", position_ids.shape, i64_bytes(position_ids)),
        ("out_cos", "bfloat16", cos.shape, bf16_bytes(cos)),
        ("out_sin", "bfloat16", sin.shape, bf16_bytes(sin)),
    ], {"mrope_section": mrope_section, "interleaved": True,
        "rope_theta": text_cfg.rope_theta, "use_bf16_rope": False,
        "attention_scaling": rotary.attention_scaling,
        "deviation": "contract [1,32,23,128] fp32; oracle returns [1,23,128] cast to x.dtype (bf16)"})

    # rotated q/k: head-major [1,H,S,D] as eager applies RoPE on [B,H,S,D].
    q_proj = attn0.q_proj(x_gemm).view(S, NH, HD).transpose(0, 1).unsqueeze(0)  # [1,32,23,128]
    k_proj = attn0.k_proj(x_gemm).view(S, NKV, HD).transpose(0, 1).unsqueeze(0)  # [1,8,23,128]
    q_rot, k_rot = apply_rotary_pos_emb(q_proj, k_proj, cos, sin)
    writer.add("rope_rotated_q", "mrope", "B", [
        ("q", "bfloat16", q_proj.shape, bf16_bytes(q_proj)),
        ("cos", "bfloat16", cos.shape, bf16_bytes(cos)),
        ("sin", "bfloat16", sin.shape, bf16_bytes(sin)),
        ("out_q_rot", "bfloat16", q_rot.shape, bf16_bytes(q_rot)),
    ], {"apply": "q*cos + rotate_half(q)*sin", "layout": "head-major [1,H,S,D]"})
    writer.add("rope_rotated_k", "mrope", "B", [
        ("k", "bfloat16", k_proj.shape, bf16_bytes(k_proj)),
        ("cos", "bfloat16", cos.shape, bf16_bytes(cos)),
        ("sin", "bfloat16", sin.shape, bf16_bytes(sin)),
        ("out_k_rot", "bfloat16", k_rot.shape, bf16_bytes(k_rot)),
    ], {"apply": "k*cos + rotate_half(k)*sin", "layout": "head-major [1,H,S,D]"})

    # ---- attention (Class D, materialized scores) ----
    # eager inference path is head-major [B,H,S,D].
    q_n = attn0.q_norm(attn0.q_proj(x_gemm).view(S, NH, HD)).transpose(0, 1).unsqueeze(0)  # [1,32,23,128]
    k_n = attn0.k_norm(attn0.k_proj(x_gemm).view(S, NKV, HD)).transpose(0, 1).unsqueeze(0)  # [1,8,23,128]
    v_n = attn0.v_proj(x_gemm).view(S, NKV, HD).transpose(0, 1).unsqueeze(0)  # [1,8,23,128]
    q_att, k_att = apply_rotary_pos_emb(q_n, k_n, cos, sin)
    v_att = v_n
    attn_out, attn_weights = eager_attention_forward(
        attn0, q_att, k_att, v_att, attn_mask_4d,
        dropout=0.0, scaling=attn0.scaling,
    )
    # eager returns attn_output [1,23,32,128] contiguous (post hmm dim swap),
    # and attn_weights [1,32,23,23] post-softmax. Scores pre-softmax recomputed.
    key_states = k_att.repeat_interleave(GROUPS, dim=1)
    scores = torch.matmul(q_att, key_states.transpose(2, 3)) * attn0.scaling
    scores = scores + attn_mask_4d[:, :, :, : key_states.shape[-2]]
    # Contract 5.4 output layouts: scores [32,23,23], probs [32,23,23],
    # attn_out [23,32,128] (seq-major, pre-o_proj, post-transpose).
    attn_out_seq = attn_out[0]      # [23,32,128]
    scores_seq = scores[0]          # [32,23,23] pre-softmax (after scale+mask)
    probs_seq = attn_weights[0]     # [32,23,23] post-softmax
    writer.add("attn_0", "attention", "D", [
        ("q", "bfloat16", q_att.shape, bf16_bytes(q_att)),
        ("k", "bfloat16", k_att.shape, bf16_bytes(k_att)),
        ("v", "bfloat16", v_att.shape, bf16_bytes(v_att)),
        ("mask", "bfloat16", attn_mask_4d.shape, bf16_bytes(attn_mask_4d)),
        ("out_scores", "bfloat16", scores_seq.shape, bf16_bytes(scores_seq)),
        ("out_probs", "bfloat16", probs_seq.shape, bf16_bytes(probs_seq)),
        ("out_attn_out", "bfloat16", attn_out_seq.shape, bf16_bytes(attn_out_seq)),
    ], {"scaling": attn0.scaling, "groups": GROUPS, "softmax_dtype": "float32",
        "mask_add": True, "n_heads": NH, "n_kv": NKV, "head_dim": HD,
        "q_layout": "head-major [1,H,S,D] (eager)", "k_layout": "head-major [1,KV,S,D]",
        "output_layout": "contract 5.4: scores/probs [H,S,S], attn_out [S,H,D]"})

    writer.write_manifest()

    # ---- sanity checks ----
    assert torch.isfinite(scores_seq).all(), "scores contain NaN/Inf"
    assert torch.isfinite(probs_seq).all(), "probs contain NaN/Inf"
    assert torch.isfinite(attn_out_seq).all(), "attn_out contains NaN/Inf"
    # mask pattern: lower-tri+diag 0.0 (allowed), upper-tri -inf except gen rows (attend-all)
    m = attn_mask_4d[0, 0]
    gen_bool = gen_positions  # [S] bool, CUDA
    tri_low = torch.tril(torch.ones(S, S, dtype=torch.bool), diagonal=0).to(m.device)
    strict_upper = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1).to(m.device)
    non_gen_strict_upper = strict_upper & ~gen_bool.unsqueeze(1)  # strict-upper off gen rows
    assert (m[tri_low] == 0.0).all()
    assert (m[non_gen_strict_upper] == min_val).all()
    assert (m[gen_positions] == 0.0).all()
    # gen rows attend to everything, so their strict-upper entries are 0.0 too
    assert (m[gen_bool][strict_upper[gen_bool]] == 0.0).all()
    print("sanity checks OK")

    print(f"\nCaptured {len(writer.fixtures)} fixtures -> {GOLDEN_DIR}")
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402  (imported after guard in main)
    sys.exit(main())