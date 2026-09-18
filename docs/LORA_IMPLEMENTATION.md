# LoRA Implementation (o1.c)

Native HiDream-O1 LoRA support: train with the pinned musubi-tuner, apply
with a merge-on-load path (no forward-path changes) or an offline GGUF merge.

## Contract (pinned)

musubi-tuner commit `4e7c7149249e7715e9168920feb4c420423abba7` (v0.3.5),
recorded in `tools/lora/musubi.lock`.

- LoRA key prefix: `lora_unet_` + module path with `.` → `_`
- Linear layout: `down [rank, in_dim]`, `up [out_dim, rank]`
- Scale: `alpha / rank` (alpha defaults to rank when absent)
- Merge: `W' = W + multiplier * (alpha/rank) * (up @ down)`
- Phase-1 scope: **T2I, linear, Full, BF16**. Conv/I2I/unknown targets fail
  explicitly — an adapter is never partially applied.

## Toolchain (`tools/lora/`)

| Tool | Purpose |
|------|---------|
| `prepare_dataset.py` | build a musubi-tuner dataset dir + `dataset.toml` |
| `train.py` | thin wrapper around the pinned musubi-tuner (T2I linear BF16) |
| `inspect_lora.py` | validate an adapter against the canonical map |
| `build_map.py` | regenerate `config/hidream_lora_map_<profile>.json` |
| `merge_safetensors.py` | offline reference merge into a safetensors base |
| `merge_gguf.py` | offline merge into a BF16 GGUF (256B aligned, provenance) |
| `make_synthetic_fixture.py` | deterministic synthetic adapter for tests |

See `tools/lora/README.md` for the full quick start.

## Native merge-on-load

`--lora FILE[:MULT]` (repeatable, max 8) is parsed by `src/model/lora.c`,
validated against `config/hidream_lora_map_dev.json` (generated C table in
`src/model/lora_map_dev.inc`), and merged once into the resident BF16 base
weights by `src/model/lora_merge.cu`:

1. cuBLAS FP32 GEMM computes `delta = up @ down` (row-chunked, 64 MiB
   scratch budget, persistent handle, no per-tensor cudaMalloc in the hot
   path).
2. A small add/cast kernel computes `W_bf16 = bf16(float(W_bf16) + scale*delta)`.

The denoising forward is unchanged — zero LoRA overhead. The merge runs on
the store's upload stream and is synchronized before the forward starts.

## Offline GGUF merge

`merge_gguf.py` produces a BF16 GGUF with 256-byte alignment and provenance
metadata (adapter SHA-256, base revision, multiplier). The merged GGUF is
loadable by the native GGUF loader (`hd_weights_to_device_gguf`) and
produces bit-identical weights to the runtime merge.

## Verification

Test ladder (all PASS on the synthetic fixture):

1. L0 formula/naming — `tools/lora/tests/test_lora_formula.py`
2. L1 adapter validation — `inspect_lora.py` (3 targets, shapes, alpha)
3. L2 native parser — `hd_lora_apply` resolves all targets (F32/BF16/F16)
4. L3 GPU merge — runtime merge vs Python reference (bit-exact BF16)
5. L4 one real projection — q_proj parity (runtime == offline == expected)
6. L5 all targets — 257 linear targets resolve (`test_name_mapping.py`)
7. L7 runtime merge vs merged GGUF — identical weights (`test_gguf_merge.py`)
8. L8 generation — 1-step sanity PASS

Fail-closed behavior verified: unknown targets, shape mismatches, and mixed
valid/invalid adapters all abort with an explicit error and leave the base
weights untouched.

## Known limitation

Multi-step generation (>1 step) crashes with a pre-existing engine bug
("double free or corruption" in `hd_generate`'s fail path) that is unrelated
to LoRA — it reproduces on a clean checkout without any LoRA code. Tracked
separately; 1-step generation is the current end-to-end verification path.