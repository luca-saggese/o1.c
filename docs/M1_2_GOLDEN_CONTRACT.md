# M1.2 Golden Fixture Contract

This document is the **coordination contract** between the golden-capture agent
(Agent A) and the CUDA implementation agent (Agent B). Both agents MUST follow
it exactly. It is frozen before any M1.2 code is written.

## 1. Scope

M1.2 implements reference CUDA transformer primitives for the Dev checkpoint
at **V2 validation level** (deterministic fixtures, zero transformer forwards,
zero image generations). Goldens are captured ONCE from the frozen Python
oracle and reused by the C comparison harness.

## 2. Golden layout

All goldens live under `artifacts/m1/golden/` (git-ignored).

```
artifacts/m1/golden/
  manifest.json            # single index of all fixtures (see §3)
  <fixture_id>.json        # per-fixture metadata (see §4)
  <fixture_id>.bin         # raw little-endian payload(s), one file per fixture
```

`manifest.json` is a JSON object mapping `fixture_id -> {file, nbytes, sha256}`.
The C harness reads `manifest.json`, then each `<fixture_id>.json` + `.bin`.

## 3. manifest.json

```json
{
  "schema_version": 1,
  "oracle_sha": "3237a638a5c2c7be106b0175958f4c0db8c2dfbf",
  "model_profile": "dev",
  "model_revision": "b6acc2fe452b3120430620dc4354fa442ee081ea",
  "capture_date": "ISO-8601 UTC",
  "compute_dtype": "bfloat16",
  "stored_dtype": "float32",
  "seed": 42,
  "resolution": 64,
  "seq_len": 23,
  "text_seq_len": 19,
  "fixtures": {
    "<fixture_id>": {"file": "<fixture_id>.json", "nbytes": 1234, "sha256": "..."}
  }
}
```

## 4. Per-fixture JSON

```json
{
  "fixture_id": "rmsnorm_0",
  "primitive": "rmsnorm",
  "class": "B",
  "inputs": [
    {"name": "x", "file": "rmsnorm_0.bin", "offset": 0, "nbytes": 4096, "dtype": "bfloat16", "shape": [1, 23, 4096]}
  ],
  "outputs": [
    {"name": "y", "file": "rmsnorm_0.bin", "offset": 4096, "nbytes": 4096, "dtype": "bfloat16", "shape": [1, 23, 4096]}
  ],
  "params": {"eps": 1e-6, "weight_name": "language_model.layers.0.input_layernorm.weight"}
}
```

- `dtype` ∈ {`float32`, `bfloat16`}. Payloads are raw little-endian, no header.
- `offset`/`nbytes` are byte offsets into the single `.bin` file for that fixture.
- `class` is the M1 numerical contract tolerance class (A exact, B pointwise,
  C GEMM, D attention). See `docs/M1_NUMERICAL_CONTRACT.md`.
- `params` carries primitive-specific scalars/names (eps, rope_theta, groups,
  weight tensor names, etc.). The C harness uses `params` for configuration,
  never for expected values.

## 5. Fixture inventory (Agent A MUST produce all of these)

### 5.1 Class-A layout fixtures (exact, integer/boolean payloads)

| fixture_id | content | payload dtype |
|---|---|---|
| `pos_ids_t2i` | position_ids [3,1,23] int64 | int64 |
| `token_types` | token_types [1,23] int64 | int64 |
| `vinput_mask` | vinput_mask [1,23] int64 | int64 |
| `attn_mask_4d` | eager 4D mask [1,1,23,23] bf16 (0.0 / -3.3895e38) | bfloat16 |
| `mrope_sections` | mrope_section [3] int64 + interleaved flag | int64 |
| `gqa_map` | head->kv_head map [32] int64 (groups=4) | int64 |
| `head_split_merge` | q [1,23,32,128] bf16 + split [32,23,128] + merged back | bfloat16 |
| `patchify_roundtrip` | z [1,4,3072] bf16 → patches [1,4,2,2,768] → back | bfloat16 |

### 5.2 Class-B pointwise fixtures (bf16 in/out unless noted)

| fixture_id | primitive | shapes | notes |
|---|---|---|---|
| `rmsnorm_0` | rmsnorm | x [1,23,4096] → y | layer 0 input_layernorm, eps 1e-6 |
| `rmsnorm_1` | rmsnorm | x [1,23,4096] → y | layer 0 post_attention_layernorm |
| `qnorm_0` | rmsnorm | x [23,32,128] → y | q_norm head_dim 128 |
| `k_norm_0` | rmsnorm | x [23,8,128] → y | k_norm head_dim 128 |
| `silu_0` | silu | x [1,23,12288] → y | elementwise |
| `swiglu_0` | swiglu | gate [1,23,12288], up [1,23,12288] → y | silu(gate)*up |
| `rope_cos_sin` | mrope | position_ids [3,1,23] → cos [1,32,23,128], sin | USE_BF16_ROPE=0 → fp32 |
| `rope_rotated_q` | mrope | q [23,32,128] + pos → rotated q | apply_rotary_pos_emb |
| `rope_rotated_k` | mrope | k [23,8,128] + pos → rotated k | |
| `t_emb_0` | timestep_embed | t [1] int64 → [1,4096] | timestep_embedding |
| `x_embed_0` | patch_embed | z [1,4,3072] → [1,23,4096] | x_embedder (incl. t_emb add) |
| `final_proj_0` | final_proj | x [1,23,4096] → [1,23,4096] | final_layer2.linear |

### 5.3 Class-C GEMM fixtures (real Dev weights, bf16 compute)

For each matmul record in `params`: `logical_input`, `weight_name`,
`weight_stored_shape`, `transpose_w`, `output_shape`, `compute_dtype`,
`accum_dtype`. Fixture payloads: input bf16, output bf16.

| fixture_id | weight | logical in → out |
|---|---|---|
| `gemm_q_proj_0` | language_model.layers.0.self_attn.q_proj.weight | [23,4096] → [23,4096] |
| `gemm_k_proj_0` | ...k_proj.weight | [23,4096] → [23,1024] |
| `gemm_v_proj_0` | ...v_proj.weight | [23,4096] → [23,1024] |
| `gemm_o_proj_0` | ...o_proj.weight | [23,4096] → [23,4096] |
| `gemm_gate_0` | ...mlp.gate_proj.weight | [23,4096] → [23,12288] |
| `gemm_up_0` | ...mlp.up_proj.weight | [23,4096] → [23,12288] |
| `gemm_down_0` | ...mlp.down_proj.weight | [23,12288] → [23,4096] |
| `gemm_t_emb_0` | t_embedder1.linear_1.weight | [1,256] → [1,4096] |
| `gemm_final_0` | final_layer2.linear.weight | [23,4096] → [23,4096] |

### 5.4 Class-D attention fixture (seq 23, materialized scores)

`attn_0` — one full eager attention step for layer 0, head 0..31, seq 23:

- inputs: q [23,32,128] bf16, k [23,8,128] bf16, v [23,8,128] bf16
  (post-norm, pre-RoPE), position_ids [3,1,23], mask [1,1,23,23] bf16.
- outputs: `scores` [32,23,23] bf16 (after scale+mask, pre-softmax),
  `probs` [32,23,23] bf16 (post-softmax), `attn_out` [23,32,128] bf16
  (pre-o_proj, post-transpose).
- params: `scaling = 128**-0.5`, `groups = 4`, `softmax_dtype = float32`.

## 6. Capture process rules (Agent A)

1. Run `tools/m1_guard.py check-env` first; abort on failure.
2. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `FA_VERSION=0`,
   `USE_BF16_ROPE=0` (must match oracle default), `sys.path.insert(0, root/python)`.
3. Load the frozen Dev model **exactly once** (single process, all fixtures).
   Use `_attn_implementation="eager"` so scores are materialized.
4. Use `torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False)` for
   all compute, matching the oracle's generation path.
5. Determinism: `torch.manual_seed(42)`, no dropout (eval mode), no RNG in
   fixtures. All fixtures derive from the canonical t2i sample
   (19 text tokens + 4 image tokens, seq 23).
6. Write payloads as raw little-endian bytes. For bf16, use
   `tensor.to(torch.bfloat16).cpu().contiguous().numpy().tobytes()`.
7. Record a ledger entry via `tools/m1_guard.py ledger` with
   `validation_level=V2`, `implementation=python-oracle-eager`,
   `whole_model_forwards=0`, `golden_id=m1.2-primitives`.
8. Do NOT run any transformer forward, denoising step, or image generation.

## 7. C harness consumption rules (Agent B)

1. Read `artifacts/m1/golden/manifest.json`, then per-fixture JSON + `.bin`.
2. Compare: shape exact, dtype exact, no NaN/Inf in candidate, then tolerance
   class from `docs/M1_NUMERICAL_CONTRACT.md`:
   - A: exact equality (integer/boolean layout).
   - B: NRMSE ≤ 2e-3 AND cosine ≥ 0.99999 (pointwise).
   - C: NRMSE ≤ 5e-3 AND cosine ≥ 0.9999 (GEMM).
   - D: NRMSE ≤ 1e-2 AND cosine ≥ 0.999 (attention).
3. For GEMM fixtures, the harness must read the real Dev weights from
   `models/dev/*.safetensors` via `src/io/safetensors.h` and apply the
   transpose/accumulation contract from `params`.
4. Class-A fixtures must match exactly (bit-for-bit for ints, exact for the
   mask pattern).

## 8. Versioning

- `schema_version: 1`. Any change to the contract bumps the version and
  requires re-capture of all goldens.
- The contract itself is versioned at `docs/M1_2_GOLDEN_CONTRACT.md`.