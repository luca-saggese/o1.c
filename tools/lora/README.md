# o1.c LoRA toolchain

Train, inspect, and apply HiDream-O1 LoRA adapters for the native o1.c
engine. The naming and formula contracts are pinned to musubi-tuner commit
`4e7c7149249e7715e9168920feb4c420423abba7` (v0.3.5) — see `musubi.lock`.

## Contract (pinned)

- LoRA key prefix: `lora_unet_` + module path with `.` → `_`
- Linear layout: `down [rank, in_dim]`, `up [out_dim, rank]`
- Scale: `alpha / rank` (alpha defaults to rank when absent)
- Merge: `W' = W + multiplier * (alpha/rank) * (up @ down)`
- Phase-1 scope: **T2I, linear, Full, BF16**. Conv/I2I/unknown targets fail
  explicitly — an adapter is never partially applied.

## Tools

| Tool | Purpose |
|------|---------|
| `prepare_dataset.py` | build a musubi-tuner dataset dir + `dataset.toml` |
| `train.py` | thin wrapper around the pinned musubi-tuner (T2I linear BF16) |
| `inspect_lora.py` | validate an adapter against the canonical map |
| `build_map.py` | regenerate `config/hidream_lora_map_<profile>.json` |
| `merge_safetensors.py` | offline reference merge into a safetensors base |
| `merge_gguf.py` | offline merge into a BF16 GGUF (256B aligned, provenance) |
| `make_synthetic_fixture.py` | deterministic synthetic adapter for tests |

## Quick start

```bash
# 1. prepare a dataset
python3 tools/lora/prepare_dataset.py \
    --input my_images/ --output datasets/my_lora \
    --width 1024 --height 1024

# 2. train (requires the pinned musubi-tuner checkout)
python3 tools/lora/train.py \
    --dataset-config datasets/my_lora/dataset.toml \
    --pretrained-model HiDream-ai/HiDream-O1-Image-Dev-2604 \
    --output-dir out --output-name my_lora \
    --network-dim 16 --network-alpha 16 \
    --learning-rate 1e-4 --max-train-epochs 10

# 3. validate the adapter
python3 tools/lora/inspect_lora.py out/my_lora.safetensors

# 4a. native merge-on-load (no forward-path changes)
./build/hidream --prompt "..." \
    --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
    --lora out/my_lora.safetensors:1.0 --output out.png

# 4b. or offline merge into a GGUF (256B aligned, provenance metadata)
python3 tools/lora/merge_gguf.py \
    --base artifacts/models/hidream-o1-dev-bf16.gguf \
    --lora out/my_lora.safetensors:1.0 \
    --output artifacts/models/hidream-o1-dev-my_lora-bf16.gguf \
    --map config/hidream_lora_map_dev.json
```

## Native merge-on-load

`--lora FILE[:MULT]` (repeatable, max 8) is parsed by `src/model/lora.c`,
validated against `config/hidream_lora_map_dev.json` (generated C table in
`src/model/lora_map_dev.inc`), and merged once into the resident BF16 base
weights by `src/model/lora_merge.cu` (cuBLAS FP32 `up @ down`, then an
add/cast kernel). The denoising forward is unchanged — zero LoRA overhead.

## Test ladder

1. L0 formula/naming — `tools/lora/tests/test_lora_formula.py`
2. L1 adapter validation — `inspect_lora.py` on the synthetic fixture
3. L2 native parser — `hd_lora_apply` resolves all targets
4. L3 GPU merge — runtime merge vs Python reference (bit-exact BF16)
5. L4 one real projection — q_proj parity (runtime == offline == expected)
6. L5 all targets — 257 linear targets resolve
7. L7 runtime merge vs merged GGUF — identical weights
8. L8 generation — 1-step sanity (multi-step blocked by a pre-existing
   engine crash, tracked separately)