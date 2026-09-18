# HiDream-O1 LoRA support for `o1.c`
## Complete implementation plan: training, native application, and GGUF materialization

**Status:** implementation specification  
**Target:** `o1.c` / HiDream-O1 Image  
**Primary deployment:** NVIDIA GB10, BF16, native C/CUDA  
**Date:** 2026-09-17

---

## 1. Goal

Add first-class LoRA support to `o1.c` while preserving the current runtime rule: **Python is allowed for offline authoring/training/conversion, but never required for production inference**.

The implementation must provide four capabilities:

1. Train a HiDream-O1 LoRA through a reproducible Python workflow based on pinned `musubi-tuner`.
2. Inspect and validate the resulting `.safetensors` adapter and map every LoRA tensor to the exact HiDream-O1 base weight used by `o1.c`.
3. Apply a LoRA directly in `o1.c` by **merging it once into resident base weights at model load**, so the denoising forward remains unchanged.
4. Materialize one or more LoRAs into a BF16 GGUF for deployment, so the deployed artifact has no runtime adapter overhead at all.

Target flow:

```text
training images + captions
        |
        v
musubi-tuner (offline Python)
        |
        v
adapter.safetensors
        |
        +---------------------------+
        |                           |
        v                           v
o1.c --lora                 tools/lora/merge_gguf.py
GPU merge-on-load                  |
        |                           v
        v                   merged BF16 GGUF
native generation                  |
                                    v
                               native generation
```

The preferred production artifact is the merged GGUF. Runtime `.safetensors` LoRA support exists for experimentation and rapid iteration.

---

## 2. Upstream contracts

Current `musubi-tuner` has explicit HiDream-O1 support.

Training uses:

```text
hidream_o1_train_network.py
--network_module networks.lora_hidream_o1
```

For T2I, current upstream targets the Qwen3-VL decoder and the HiDream pixel input/output layers. Token embeddings and the LM head are excluded. I2I additionally targets Qwen3-VL visual modules.

For a linear LoRA, musubi uses:

```text
lora_down.weight : [rank, in_dim]
lora_up.weight   : [out_dim, rank]
alpha            : scalar
```

with:

```text
scale = alpha / rank

W_merged =
    W_base +
    multiplier * (lora_up @ lora_down) * scale
```

This formula is the compatibility contract for `o1.c`.

Musubi forms LoRA module names by prefixing the PyTorch module path and replacing `.` with `_`. `o1.c` must **not** reverse underscores heuristically. Mapping must always be built from a known base module path toward the expected LoRA name, or through a generated explicit map.

GGUF v3 supports one-file storage, extensible metadata, per-tensor descriptors, `general.alignment`, and BF16 (`GGML_TYPE_BF16`). For `o1.c` use:

```text
GGUF version:       3
general.alignment:  256
general.architecture: "hidreamo1"
```

---

## 3. Scope

### Phase 1 — required

Support:

```text
T2I LoRA
linear layers
BF16 base weights
musubi LoRA .safetensors
BF16/F16/F32 adapter tensors
single LoRA
multiple LoRAs, sequentially applied
runtime merge-on-load
offline safetensors merge for debugging
offline GGUF merge for deployment
Full model
Dev application when profile/shapes match
```

### Explicit Phase-1 exclusions

Do not block Phase 1 on:

```text
runtime unmerged LoRA forward hooks
per-token LoRA scaling
adapter hot-swap without restoring base weights
Conv2d / Conv3d LoRA
I2I visual-tower LoRA
DoRA / LoHa / LoKr
quantized LoRA
FP8/NVFP4 base merge
training implemented in C/CUDA
```

If an adapter contains unsupported target types, strict mode must fail. Never silently pretend the adapter was fully applied.

---

# Part A — Repository layout

## 4. Proposed tree

```text
o1.c/
├── tools/
│   └── lora/
│       ├── README.md
│       ├── requirements.txt
│       ├── musubi.lock
│       ├── prepare_dataset.py
│       ├── train.py
│       ├── inspect.py
│       ├── build_map.py
│       ├── merge_safetensors.py
│       ├── merge_gguf.py
│       ├── gguf.py
│       ├── common.py
│       ├── configs/
│       │   ├── dataset.example.toml
│       │   ├── train_full_t2i.example.toml
│       │   └── sample_prompts.example.toml
│       └── tests/
│           ├── test_lora_formula.py
│           ├── test_name_mapping.py
│           └── test_gguf_merge.py
│
├── config/
│   ├── hidream_lora_map_full.json
│   └── hidream_lora_map_dev.json
│
├── include/
│   └── hd_lora.h
│
├── src/
│   └── model/
│       ├── lora.c
│       └── lora_merge.cu
│
├── tests/
│   └── unit/
│       ├── test_lora_loader.c
│       └── test_lora_merge.cu
│
└── docs/
    └── LORA_IMPLEMENTATION.md
```

The public GitHub tree may lag the current local M2 branch. Adapt exact filenames to the local tree, but preserve these ownership boundaries:

```text
tools/lora/      Python authoring/conversion only
src/model/       model-level LoRA parsing/lifecycle
src/cuda or .cu  GPU merge implementation
config/          deterministic name maps
docs/            operational and numerical contracts
```

---

# Part B — Python training workflow

## 5. Dependency policy

Do not reimplement the trainer inside `o1.c`.

`tools/lora/train.py` is an orchestration wrapper over a **pinned** external `musubi-tuner`.

`musubi.lock`:

```text
repository=https://github.com/kohya-ss/musubi-tuner.git
commit=<IMMUTABLE_TESTED_SHA>
```

The implementation commit must replace the placeholder with an actually tested SHA. Never depend on moving `main`.

Minimal local requirements:

```text
safetensors>=0.5
numpy>=2.0
Pillow>=10
tomli>=2.0; python_version < "3.11"
```

Musubi owns its own PyTorch/Accelerate/optimizer dependencies.

---

## 6. Dataset workspace

Recommended layout:

```text
artifacts/lora/<name>/
├── dataset/
│   ├── 0001.png
│   ├── 0001.txt
│   ├── 0002.png
│   ├── 0002.txt
│   └── ...
├── cache/
├── checkpoints/
├── samples/
├── dataset.toml
├── train.json
└── manifest.json
```

`artifacts/` remains git-ignored.

Caption files use the same basename:

```text
0001.png
0001.txt
```

A trigger token is merely a caption/training convention. It is not a new tokenizer special token.

---

## 7. `prepare_dataset.py`

Purpose:

- validate image/caption pairs;
- optionally copy or symlink data into a deterministic workspace;
- validate UTF-8 captions;
- verify image readability;
- emit `dataset.toml`;
- write image/caption SHA256 hashes;
- reject duplicates/orphans.

CLI:

```bash
python tools/lora/prepare_dataset.py \
  --images /data/my_subject \
  --output artifacts/lora/my_subject \
  --resolution 1024 \
  --caption-ext .txt \
  --bucket \
  --no-upscale
```

Generated example:

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "/absolute/path/artifacts/lora/my_subject/dataset"
cache_directory = "/absolute/path/artifacts/lora/my_subject/cache"
num_repeats = 1
```

Validation:

```text
>= 1 image
caption for every image
no orphan caption
valid image dimensions
UTF-8 caption
canonical path unique
SHA256 recorded
```

Do not automatically rewrite captions.

---

## 8. `train.py`

### Responsibilities

`train.py` provides a stable project-owned interface while delegating math/training to pinned musubi.

Sequence:

```text
verify musubi pin
verify base model
verify dataset
cache pixel patch tokens
cache prompt token IDs
launch LoRA training
inspect final adapter
write reproducibility manifest
```

### CLI

```bash
python tools/lora/train.py \
  --musubi external/musubi-tuner \
  --model models/full/hidream_o1_image_bf16.safetensors \
  --model-type full \
  --task t2i \
  --dataset artifacts/lora/my_subject/dataset.toml \
  --output artifacts/lora/my_subject/checkpoints \
  --name my_subject \
  --rank 32 \
  --alpha 32 \
  --learning-rate 4e-5 \
  --epochs 16 \
  --seed 42 \
  --flash-attn \
  --blocks-to-swap 24
```

### Full/T2I command generated by wrapper

```bash
accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  src/musubi_tuner/hidream_o1_train_network.py \
  --dit /abs/path/hidream_o1_image_bf16.safetensors \
  --dataset_config /abs/path/dataset.toml \
  --model_type full \
  --task t2i \
  --mixed_precision bf16 \
  --timestep_sampling uniform \
  --weighting_scheme none \
  --noise_scale_start 8.0 \
  --noise_scale_end 8.0 \
  --noise_clip_std 0.0 \
  --optimizer_type adamw8bit \
  --learning_rate 4e-5 \
  --gradient_checkpointing \
  --network_module networks.lora_hidream_o1 \
  --network_dim 32 \
  --network_alpha 32 \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --output_dir /abs/path/checkpoints \
  --output_name my_subject
```

Optional memory/performance flags:

```text
--flash_attn
--blocks_to_swap 24
--use_pinned_memory_for_block_swap
--skip_t2i_visual_dummy
```

For single-GPU T2I, upstream documents `--skip_t2i_visual_dummy` as a numerically neutral optimization because the dummy visual pass is only needed to keep distributed collectives symmetric.

### Dev profile

If training Dev, choose as one coherent profile:

```text
--model_type dev
--noise_scale_start 7.5
--noise_scale_end 7.5
--noise_clip_std 2.5
```

Do not permit a Full/Dev mismatch without an explicit expert override.

---

## 9. Cache stages

Run pixel-token cache:

```bash
python src/musubi_tuner/hidream_o1_cache_pixel.py \
  --dataset_config /abs/path/dataset.toml \
  --batch_size 1
```

Then prompt-token cache:

```bash
python src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py \
  --dataset_config /abs/path/dataset.toml \
  --batch_size 16
```

HiDream-O1 here does not use a VAE latent cache; the image cache contains normalized 32x32 pixel patch tokens.

---

## 10. Training manifest

Write:

```text
artifacts/lora/<name>/manifest.json
```

Example:

```json
{
  "format": "o1-lora-training-manifest-v1",
  "created_at": "...",
  "base_model_type": "full",
  "base_model_path": "...",
  "base_model_sha256": "...",
  "musubi_repository": "https://github.com/kohya-ss/musubi-tuner.git",
  "musubi_commit": "...",
  "task": "t2i",
  "rank": 32,
  "alpha": 32.0,
  "learning_rate": 0.00004,
  "epochs": 16,
  "seed": 42,
  "dataset_manifest_sha256": "...",
  "adapter_path": "...",
  "adapter_sha256": "..."
}
```

This is evidence/provenance, not a production runtime dependency.

---

# Part C — Adapter inspection and mapping

## 11. Naming contract

Musubi builds a LoRA prefix from a known module path:

```text
module path:
model.layers.0.self_attn.q_proj

expected prefix:
lora_unet_model_layers_0_self_attn_q_proj
```

The actual current HiDream names must be derived from the pinned model; the path above is illustrative.

Never implement:

```text
replace("_", ".")
```

because it is not reversible.

Instead:

```text
known base module path
    -> prefix + module_path.replace(".", "_")
    -> exact LoRA key lookup
```

---

## 12. `build_map.py`

Generate a committed mapping from the actual tensor manifest.

CLI:

```bash
python tools/lora/build_map.py \
  --tensor-manifest config/tensor_manifest_full.json \
  --profile full \
  --output config/hidream_lora_map_full.json
```

Entry shape:

```json
{
  "base_tensor": "actual.upstream.path.q_proj.weight",
  "base_module": "actual.upstream.path.q_proj",
  "lora_prefix": "lora_unet_actual_upstream_path_q_proj",
  "kind": "linear",
  "shape": [4096, 4096],
  "o1_role": "block.0.q_proj",
  "gguf_name": "blk.00.attn.q.weight"
}
```

The mapping becomes the explicit compatibility boundary among:

```text
musubi name
upstream state-dict name
o1 internal weight role
GGUF tensor name
```

---

## 13. `inspect.py`

CLI:

```bash
python tools/lora/inspect.py \
  --lora artifacts/lora/my_subject/checkpoints/my_subject.safetensors \
  --base-manifest config/tensor_manifest_full.json \
  --map config/hidream_lora_map_full.json \
  --strict
```

Report:

```text
LoRA: my_subject.safetensors
format: musubi LoRA
task compatibility: T2I linear
modules: ...
rank distribution: ...
alpha distribution: ...

supported:
  linear: ...

unsupported:
  conv: ...
  unknown: ...

base profile: full
shape validation: PASS
mapping validation: PASS
```

Linear validation:

```text
down.shape == [rank, in_dim]
up.shape   == [out_dim, rank]
base.shape == [out_dim, in_dim]
```

Reject:

```text
missing up/down pair
rank mismatch
invalid alpha tensor
unknown target
missing base tensor
shape mismatch
NaN / Inf
unsupported Conv entry in strict Phase-1 mode
```

If alpha is absent, use musubi semantics:

```text
alpha = rank
scale = 1
```

---

# Part D — Native `o1.c` support

## 14. Core decision: merge on load

Do **not** initially execute:

```text
Y = XW^T + s*(XA^T)B^T
```

inside every denoising step.

That would add two GEMMs per adapted linear and undo part of the M2 performance gain.

Use:

```text
load base weights
     |
load LoRA
     |
GPU merge once
     |
same BF16 weight pointers
     |
unchanged cuBLAS/cudnn forward
```

After merge, the forward has zero LoRA-specific work.

---

## 15. CLI

Add repeatable:

```text
--lora FILE[:MULTIPLIER]
```

Example:

```bash
./build/hidream \
  --model artifacts/models/hidream-o1-full-bf16.gguf \
  --lora portrait.safetensors:0.8 \
  --lora style.safetensors:0.35 \
  --prompt "..."
```

Default multiplier:

```text
1.0
```

If Windows support later makes colon ambiguous with drive letters, add an alternative structured flag instead of clever parsing.

---

## 16. C API

Prefer an explicit per-model configuration, never a hidden global.

Concept:

```c
typedef struct {
    const char *path;
    float multiplier;
} hd_lora_spec;

typedef struct {
    const hd_lora_spec *items;
    size_t count;
} hd_lora_config;
```

Either extend load options or expose:

```c
int hd_model_apply_lora(
    hd_model *model,
    const char *path,
    float multiplier,
    hd_error *err);
```

Contract:

```text
model must be idle
apply before generation
operation is synchronous from caller view
weight pointers remain stable
no new forward allocation
```

Changing/removing LoRA in Phase 1 requires reloading the pristine base.

---

## 17. Native structures

Reuse existing tensor/weight structures where possible.

Conceptual internal representation:

```c
typedef enum {
    HD_LORA_LINEAR = 1,
    HD_LORA_CONV   = 2,
} hd_lora_kind;

typedef struct {
    const char *lora_prefix;
    const char *base_tensor_name;

    uint32_t rank;
    uint32_t in_dim;
    uint32_t out_dim;

    float alpha;
    float multiplier;

    hd_tensor_view down;
    hd_tensor_view up;
    hd_tensor_view *base;
} hd_lora_entry;

typedef struct {
    hd_lora_entry *entries;
    size_t count;
    char sha256[65];
    char base_profile[16];
} hd_lora_adapter;
```

---

## 18. Reuse the safetensors parser

Do not add a second parser.

Reuse the existing `o1.c` safetensors infrastructure for:

```text
header/table parsing
dtype
shape
file offset
metadata
bounded reads
```

Accepted Phase-1 adapter tensor dtypes:

```text
BF16
F16
F32
```

Merge compute is FP32.

Reject quantized adapter tensors initially.

---

# Part E — GPU merge

## 19. Formula

For a base linear:

```text
W : [N,K]
A = down : [R,K]
B = up   : [N,R]

s = user_multiplier * alpha/R

W' = W + s*(B@A)
```

The resulting stored layout must be exactly the layout expected by current `hd_linear`/cuBLAS.

---

## 20. Bounded scratch

Do not materialize the entire FP32 delta for a huge matrix.

Use output-row chunks.

For chunk `row0:row1`:

```text
B_chunk = B[row0:row1,:]      [C,R]
delta   = B_chunk @ A           [C,K] FP32
W_chunk = BF16(FP32(W_chunk) + s*delta)
```

Choose `C` from a scratch-memory budget, e.g. 64 MiB, not from a fixed magic number.

---

## 21. GPU sequence

For every target:

```text
validate W/A/B
upload or bind A
upload/stream B

for each row chunk:
    cuBLAS FP32-compute GEMM:
        delta = B_chunk @ A

    CUDA add/cast:
        W_bf16 = bf16(float(W_bf16) + scale*delta_fp32)
```

No change to `hd_linear`.

Conceptual add kernel:

```c
__global__ void hd_lora_add_bf16(
    __nv_bfloat16 *w,
    const float *delta,
    size_t n,
    float scale)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float base = __bfloat162float(w[i]);
        w[i] = __float2bfloat16_rn(base + scale * delta[i]);
    }
}
```

Use the project-standard BF16 conversion primitive if one already exists.

---

## 22. cuBLAS policy

This is load-time, not hot-path compute. Prefer correctness.

```text
A/B:       original adapter dtype or canonical converted staging dtype
compute:   FP32
delta:     FP32
```

Freeze a deterministic tiny matrix test for row-major semantics. Do not copy unverified transpose logic from another call site.

---

## 23. Multiple LoRAs

Define Phase-1 semantics:

```text
LoRAs are applied in CLI order.
After each adapter, the resident weight is BF16.
```

Thus:

```text
W0 = base
W1 = BF16(W0 + s1*B1A1)
W2 = BF16(W1 + s2*B2A2)
```

Sequential BF16 rounding means order is technically observable.

The GGUF merger must use the same `sequential-bf16-v1` contract if exact native/merged parity is desired.

A later `accumulate-all-deltas-in-FP32` mode would be a different contract and must be versioned.

---

## 24. Failure atomicity

Before mutating any base weight:

```text
parse entire adapter
resolve every target
validate all shapes
validate all dtypes
verify supported kinds
allocate required scratch
```

Only then merge.

If CUDA fails mid-merge:

```text
mark model invalid
require reload
do not generate with partially merged weights
```

---

# Part F — Native name mapping

## 25. No string lookup in forward

Resolve names only during load.

Preferred implementation: generate a compact C table from the JSON mapping.

Concept:

```c
static const hd_lora_map_entry k_full_lora_map[] = {
    { "lora_unet_...", "actual.base.tensor.weight", HD_WEIGHT_BLOCK_Q, 0 },
    ...
};
```

Final binding must resolve to existing pre-bound weight pointers.

Runtime forward does not carry LoRA names.

---

# Part G — Offline safetensors merge

## 26. `merge_safetensors.py`

Useful for debugging/parity even if not used for final deployment.

```bash
python tools/lora/merge_safetensors.py \
  --base models/full/hidream_o1_image_bf16.safetensors \
  --lora portrait.safetensors:0.8 \
  --output artifacts/lora/portrait/hidream_full_portrait_bf16.safetensors \
  --map config/hidream_lora_map_full.json
```

Process one tensor at a time:

```text
untouched -> preserve/copy
adapted   -> FP32(base) + scale*(up@down) -> BF16
```

Do not load the whole model into host RAM if streaming/safe-open can avoid it.

Uses:

```text
reference merge
interop
musubi comparison
native merge verification
```

---

# Part H — GGUF format

## 27. Required metadata

Use:

```text
GGUF v3
general.architecture = "hidreamo1"
general.alignment    = 256
BF16                  = GGML_TYPE_BF16
```

Keep tensor names compact.

Recommended namespaces:

```text
tok_emb.weight

blk.00.attn.q.weight
blk.00.attn.k.weight
blk.00.attn.v.weight
blk.00.attn.o.weight

blk.00.ffn.gate.weight
blk.00.ffn.up.weight
blk.00.ffn.down.weight

...
final.norm.weight
final.proj.weight

pix.in.*
pix.out.*
time.*
vis.blk.00.*
```

The exact mapping must come from the real model manifest.

---

## 28. Provenance metadata for merged GGUF

A merged file is not the pristine base model.

Record:

```text
o1.format.version = 1
o1.model.profile = "full"
o1.weights.dtype = "bf16"

o1.base.sha256 = "..."
o1.base.source = "HiDream-O1-Image"

o1.lora.count = 1
o1.lora.0.name = "portrait"
o1.lora.0.sha256 = "..."
o1.lora.0.multiplier = 0.8
o1.lora.0.merge = "sequential-bf16-v1"

o1.tensor_map.sha256 = "..."
```

For multiple adapters use indexed metadata in application order.

---

# Part I — Minimal GGUF helper

## 29. `gguf.py`

Implement only the subset required by the project:

```text
read/write magic/version
read/write KV values
read/write tensor descriptors
calculate aligned tensor-data offset
calculate aligned tensor offsets
BF16 type
raw payload copy
```

Validation:

```text
magic == GGUF
version == 3
little endian supported
alignment power of two
alignment >= 8
tensor offsets aligned
payload ranges inside file
no duplicate tensor name
supported tensor type
```

No need to import the full ggml runtime.

---

# Part J — Merge directly into GGUF

## 30. `merge_gguf.py`

Preferred deployment path.

```bash
python tools/lora/merge_gguf.py \
  --base artifacts/models/hidream-o1-full-bf16.gguf \
  --lora artifacts/lora/portrait/checkpoints/portrait.safetensors:0.8 \
  --output artifacts/models/hidream-o1-full-portrait-bf16.gguf \
  --map config/hidream_lora_map_full.json \
  --device cuda
```

Multiple:

```bash
python tools/lora/merge_gguf.py \
  --base artifacts/models/hidream-o1-full-bf16.gguf \
  --lora portrait.safetensors:0.8 \
  --lora style.safetensors:0.35 \
  --output artifacts/models/hidream-o1-custom-bf16.gguf \
  --map config/hidream_lora_map_full.json \
  --device cuda
```

---

## 31. GGUF merge algorithm

### Pass 1 — validate

Parse:

```text
base metadata
base tensor descriptors
LoRA descriptors
mapping
```

Resolve all adapter targets before writing anything.

### Pass 2 — build output descriptors

Preserve:

```text
tensor order
tensor names
tensor shapes
tensor dtypes
```

unless this run is explicitly a format migration.

Recalculate offsets with:

```text
alignment = 256
```

### Pass 3 — payload

For untouched tensors:

```text
raw-copy exact payload bytes
```

Do not decode/re-encode BF16.

For adapted tensors:

```text
read base BF16
apply sequential-bf16-v1 merge
write new BF16 payload
```

Use row-chunked GPU merge if needed.

### Finalization

Write:

```text
output.tmp
```

then:

```text
re-open
validate descriptor/range integrity
compute SHA256
write sidecar manifest
atomic rename
```

Never modify the base GGUF in place.

---

# Part K — Optional one-pass safetensors -> GGUF + LoRA

## 32. Combined converter

After the standalone base GGUF converter and LoRA merge are independently proven, optionally support:

```bash
python tools/lora/merge_gguf.py \
  --base-safetensors models/full/hidream_o1_image_bf16.safetensors \
  --lora portrait.safetensors:0.8 \
  --output artifacts/models/hidream-o1-full-portrait-bf16.gguf \
  --profile full
```

Flow:

```text
safetensors
 -> canonical o1 tensor map
 -> optional LoRA merge
 -> GGUF
```

Do not implement the combined path before both individual transformations have tests.

---

# Part L — Fast loader interaction

## 33. Target GGUF loading path

```text
open GGUF
   |
parse metadata/tensor table
   |
calculate final CUDA arena
   |
one large aligned CUDA allocation
   |
large sequential file reads
   |
2-4 persistent pinned host staging buffers
   |
cudaMemcpyAsync to final offsets
   |
one final synchronization
   |
pre-bind weight pointers
```

If raw LoRA supplied:

```text
base upload
   |
LoRA validate
   |
GPU merge in-place
   |
free LoRA staging
   |
normal generation
```

If merged GGUF supplied:

```text
GGUF upload
   |
normal generation
```

This is why GGUF merge is the preferred deployment route.

---

# Part M — Compatibility and metadata

## 34. Base compatibility

Validate at minimum:

```text
profile
target names
target shapes
```

If an `o1` sidecar exists, also validate:

```text
base SHA256
training model revision
mapping hash
```

A raw third-party LoRA may not contain enough provenance to prove exact base revision. In that case state:

```text
shape/name compatibility verified
exact base revision unknown
```

Option:

```text
--lora-strict-base
```

requires exact known base hash.

---

## 35. Scale semantics

Always:

```text
effective_scale =
    user_multiplier * alpha/rank
```

Default multiplier:

```text
1.0
```

Do not reinterpret alpha.

---

## 36. Adapter sidecar

`inspect.py` should be able to emit:

```text
portrait.o1lora.json
```

Example:

```json
{
  "format": "o1-lora-v1",
  "adapter_sha256": "...",
  "base_model_sha256": "...",
  "profile": "full",
  "task": "t2i",
  "producer": "musubi-tuner",
  "producer_commit": "...",
  "contains_conv": false,
  "supported_by_o1": true,
  "mapping_sha256": "...",
  "entries": 252
}
```

The native loader may use it for stronger checks but must still be able to inspect a standard musubi `.safetensors`.

---

# Part N — Test ladder

## 37. L0 — Formula

Synthetic:

```text
N=3
K=5
R=2
```

with known non-zero values.

Verify exactly:

```text
W' = W + multiplier*(alpha/R)*(B@A)
```

### L1 — Name mapping

Frozen adapter fixture:

```text
every known prefix resolves once
all up/down pairs complete
no ambiguity
```

### L2 — Native parser

Validate:

```text
name
dtype
shape
rank
alpha
file offset
```

### L3 — Native tiny GPU merge

Use BF16-exact synthetic values where possible and require exact BF16 bytes. Otherwise use a predefined numerical contract.

### L4 — One real projection

Freeze one actual `q_proj` or similar:

```text
base W
down
up
alpha
multiplier
Python merged W
```

Compare native:

```text
NRMSE
cosine
max_abs
```

Freeze thresholds from the first correct implementation; do not relax them ad hoc.

### L5 — All targets

Python and native must resolve the same:

```text
target count
ordered target list
rank distribution
```

### L6 — Python merge semantics

Compare in musubi:

```text
base + --lora_weight
```

versus an offline merged base with identical generation settings.

### L7 — Native runtime merge vs merged GGUF

Compare:

```text
A. base GGUF + --lora adapter:scale
B. pre-merged GGUF
```

under the exact same `sequential-bf16-v1` contract.

### L8 — Image sanity

Fixed:

```text
prompt
seed
resolution
steps
scheduler
LoRA multiplier
```

Generate:

```text
base
runtime-LoRA
merged-GGUF-LoRA
```

Runtime and merged GGUF should satisfy the normal project determinism/numerical contract.

---

# Part O — Performance

## 38. Forward requirement

After merge:

```text
LoRA-specific forward overhead = 0
```

No extra GEMM inside denoising.

### Load instrumentation

Under existing compile-time timing only:

```text
BASE_LOAD_TOTAL
LORA_PARSE
LORA_UPLOAD
LORA_MERGE
LORA_TOTAL
```

No debug synchronization in production.

For pre-merged GGUF:

```text
startup ~= normal GGUF startup
```

with no adapter merge stage.

---

# Part P — User workflows

## 39. Train

```bash
python tools/lora/train.py \
  --musubi external/musubi-tuner \
  --model models/full/hidream_o1_image_bf16.safetensors \
  --model-type full \
  --dataset artifacts/lora/luca/dataset.toml \
  --name luca \
  --rank 32 \
  --alpha 32 \
  --learning-rate 4e-5 \
  --epochs 16 \
  --seed 42
```

## 40. Inspect

```bash
python tools/lora/inspect.py \
  --lora artifacts/lora/luca/checkpoints/luca.safetensors \
  --map config/hidream_lora_map_full.json \
  --strict
```

## 41. Native rapid use

```bash
./build/hidream \
  --model artifacts/models/hidream-o1-full-bf16.gguf \
  --lora artifacts/lora/luca/checkpoints/luca.safetensors:0.8 \
  --prompt "portrait of ohwx person in dramatic studio light"
```

## 42. Materialize deployment GGUF

```bash
python tools/lora/merge_gguf.py \
  --base artifacts/models/hidream-o1-full-bf16.gguf \
  --lora artifacts/lora/luca/checkpoints/luca.safetensors:0.8 \
  --output artifacts/models/hidream-o1-full-luca-bf16.gguf \
  --map config/hidream_lora_map_full.json \
  --device cuda
```

Then:

```bash
./build/hidream \
  --model artifacts/models/hidream-o1-full-luca-bf16.gguf \
  --prompt "portrait of ohwx person in dramatic studio light"
```

---

# Part Q — Errors and robustness

## 43. Actionable errors

Examples:

```text
LoRA error: unsupported Conv3d target
key: ...
Phase-1 o1.c supports linear T2I LoRA only.
```

```text
LoRA error: shape mismatch
base: ... [N,K]
down: ... [R,K]
up:   ... [wrong_N,R]
```

```text
LoRA error: target unknown for profile 'dev'
prefix: ...
```

```text
LoRA error: base hash mismatch
adapter base: ...
loaded base:  ...
```

Avoid generic `"invalid adapter"` without context.

---

## 44. Treat files as untrusted binary input

Check overflow for:

```text
shape products
element counts
offset + length
alignment calculations
arena sizes
```

Reject:

```text
rank 0
malformed scalar alpha
NaN/Inf alpha or multiplier
duplicate forbidden keys
out-of-range offsets
unsupported dtype
overlapping malformed ranges
```

No allocation based on unchecked file-controlled multiplication.

---

# Part R — I2I / visual LoRA follow-up

## 45. Why separate

Current musubi HiDream support can train I2I adapters and may include Qwen3-VL visual modules. Non-1x1 convolution LoRA is included when `conv_dim` is supplied.

That needs convolution-specific merge semantics rather than the simple linear:

```text
up @ down
```

Phase 1 must detect such entries and fail clearly.

Do not apply only the linear subset of an I2I adapter and report success.

---

# Part S — Commit plan

## 46. Suggested commits

```text
tools(lora): add pinned HiDream LoRA training workflow
tools(lora): add adapter inspection and canonical mapping
feat(lora): load and validate musubi adapters
feat(lora): merge linear adapters on GPU at model load
feat(cli): add repeatable --lora path:multiplier
feat(gguf): materialize LoRA into BF16 GGUF
test(lora): freeze runtime/GGUF parity evidence
docs(lora): document training and deployment
```

Each commit should leave:

```bash
git status
```

clean.

---

# Part T — Definition of done

## 47. Checklist

- [ ] musubi is pinned to immutable tested SHA
- [ ] dataset preparation script works
- [ ] `train.py` produces a Full T2I LoRA
- [ ] training manifest contains base/tool hashes
- [ ] `inspect.py` resolves all adapter entries
- [ ] Full canonical mapping is committed
- [ ] Dev mapping exists if Dev support is advertised
- [ ] native reader accepts BF16/F16/F32 adapter tensors
- [ ] unsupported conv/I2I entries fail explicitly
- [ ] GPU merge matches musubi formula
- [ ] no LoRA operation remains in forward after merge
- [ ] multiple adapters apply deterministically in CLI order
- [ ] `--lora path:scale` works
- [ ] runtime merge matches pre-merged GGUF under frozen contract
- [ ] merged GGUF records provenance
- [ ] merged GGUF uses 256-byte alignment
- [ ] untouched tensor payloads are raw-copied
- [ ] GGUF output is atomically created and revalidated
- [ ] production build has no temporary debug sync/logging
- [ ] generation performance after merge matches base within noise
- [ ] complete train -> inspect -> run -> merge -> deploy instructions exist
- [ ] repository is clean

---

# Part U — Recommended first version

## 48. Minimal useful implementation

```text
TRAIN
  musubi Full T2I
  rank 32 / alpha 32
  linear adapters

IMPORT
  standard musubi .safetensors
  strict names/shapes

NATIVE
  GPU merge-on-load
  FP32 delta
  BF16 resident result
  zero forward overhead

DEPLOY
  BF16 GGUF
  256-byte alignment
  pre-merged adapter
  provenance metadata
```

Do not start with a dynamic LoRA forward.

The feature should preserve the defining property of the current M2 engine:

```text
the production denoising forward remains simple and fast
```

---

# Part V — Numerical policy

## 49. Keep LoRA parity separate from the known M2 cuBLAS discrepancy

The current production cuBLAS path has a small known numerical discrepancy from the historical reference path.

Do not use that as a reason to weaken LoRA validation.

Separate:

```text
1. LoRA merge math
   Python merged weights vs native merged weights

2. LoRA deployment
   native runtime merge vs pre-merged GGUF

3. End-to-end model
   existing o1.c numerical contract
```

First prove the resident weights are correct. Only then test image generation.

---

# Part W — Agent execution order

## 50. Do this in sequence

```text
1. Inspect the current local o1.c tree and adjust paths.
2. Pin a tested musubi commit.
3. Implement dataset/training wrapper.
4. Produce one tiny real rank-4/rank-8 LoRA fixture.
5. Inspect exact produced safetensors keys.
6. Generate canonical Full mapping from actual model manifest.
7. Freeze a deterministic synthetic merge fixture.
8. Implement native parser/validator.
9. Implement one-linear GPU merge.
10. Pass tiny synthetic test.
11. Pass one real q_proj test.
12. Implement all supported linear mappings.
13. Add CLI.
14. Compare native merged weights with Python reference.
15. Implement/validate GGUF reader-writer support if still pending.
16. Implement LoRA merge into GGUF.
17. Require runtime-merge vs merged-GGUF parity.
18. Benchmark load overhead.
19. Run one fixed-seed image test.
20. Remove temporary debug.
21. Update docs.
22. Commit.
23. Verify git status is clean.
```

If mapping is ambiguous, fail. Do not guess.

---

# External references

Implementation must be rechecked against the exact pinned revisions when coding:

- `kohya-ss/musubi-tuner/docs/hidream_o1.md`
  - HiDream-O1 caching, T2I/I2I LoRA training, Full/Dev noise settings, inference with LoRA.
- `kohya-ss/musubi-tuner/src/musubi_tuner/networks/lora_hidream_o1.py`
  - HiDream-specific target classes and exclusion of embeddings/LM head.
- `kohya-ss/musubi-tuner/src/musubi_tuner/networks/lora.py`
  - LoRA down/up layout, alpha/rank scaling, naming and linear merge semantics.
- `ggml-org/ggml/docs/gguf.md`
  - GGUF v3, BF16 type, metadata, tensor descriptors and global alignment.
- `ggml-org/llama.cpp/ggml/include/gguf.h`
  - implementation-facing GGUF constants and layout.

---

# Final architecture

```text
                 Python / training workstation
                           |
                      musubi-tuner
                           |
                    LoRA safetensors
                           |
             +-------------+-------------+
             |                           |
             v                           v
       o1.c merge-on-load          merge_gguf.py
             |                           |
       BF16 resident weights       merged BF16 GGUF
             |                           |
             +-------------+-------------+
                           |
                  unchanged native
                 cuBLAS/cuDNN forward
                           |
                      generated image
```

**Training belongs to the mature Python ecosystem; LoRA application belongs outside the hot path; deployment remains native, offline, and zero-Python.**
