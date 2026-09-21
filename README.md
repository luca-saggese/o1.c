# HiDream O1 Image Inference Engine (`o1.c`)

A from-scratch **native C/CUDA inference engine** for the
[HiDream O1 Image](https://huggingface.co/HiDream-ai) foundation image model,
targeted at the **NVIDIA GB10** (DGX Spark) platform: compute capability 12.1
(`-arch=sm_121`), CUDA 13.0 toolchain. The engine re-implements the model's
transformer math in hand-written CUDA kernels, validated tensor-by-tensor
against a frozen Python oracle, so that **production inference never depends
on Python**.

<div align="center">
  <video src="https://github.com/user-attachments/assets/cbbdb816-f050-4685-aa51-4741479a0e5c" width="70%" poster=""> </video>
</div>

> **HiDream-O1-Image-Dev-2604 debuts at #8 in the Artificial Analysis Text to Image Arena, which is positioned to be the new leading open weights Text to Image model.**


<p align="center">
  <img src="example_assets/general.webp" alt="General text-to-image generation" width="100%"/>
  <br><sub><b>General text-to-image generation</b> at up to 2,048 × 2,048.</sub>
</p>

<p align="center">
  <img src="example_assets/text-layout.webp" alt="Long-text rendering and layout" width="100%"/>
  <br><sub><b>Long-text rendering & layout control</b> — accurate, multi-region, multilingual text.</sub>
</p>

<p align="center">
  <img src="example_assets/IP_2.jpg" alt="Subject-driven personalization" width="100%"/>
  <br><sub><b>Subject-driven personalization</b> — preserve identity / IP across new scenes.</sub>
</p>

---

## What this project is

`o1.c` builds a deterministic, reproducible inference path for HiDream O1
Image using only C code and hand-written CUDA kernels. The production
pipeline is

```
build/hidream (CLI, src/main.c)
  -> hd_generate (src/runtime/generate.c)   sequence build, scheduler, denoise loop
  -> hd_forward  (src/model/forward.c)      transformer forward, hand-written CUDA kernels
  -> decode + PNG output (src/runtime/decode.c, src/image/)
```

Every primitive is validated against golden fixtures captured once from the
frozen official HiDream/Transformers oracle. The Python upstream is treated
strictly as an oracle — a source of truth for semantics — never as a runtime
dependency.

## Core principles

- **Python is an oracle, not a runtime.** The frozen upstream checkout lives
  in `python/` (read-only, git-ignored) and is used only for offline
  validation. Production inference is pure C/CUDA.
- **Reproducibility is contractual.** Revisions, versions, hashes, and gate
  evidence are persisted to locks and manifests (`config/`); a branch name is
  never a substitute for an immutable commit.
- **Cheap validation first.** The validation cost ladder (V0–V6) reserves
  full 28/50-step generations for milestone closure; most gates run on
  metadata, startup-only loads, or 1–3 denoising steps.
- **Dev/Base symmetry.** Dev and Base/Full profiles share one configuration
  and code path; only profile parameters differ.
- **Ownership boundaries.** Only engine code (`src/`, `tools/`, `config/`,
  `docs/`, `scripts/`, `tests/`) is versioned. Oracle checkout, model
  weights, build output, and generated artifacts are git-ignored.

## Repository layout

```
include/      Public C ABI header(s)          (hidream.h)
src/          C/CUDA engine sources
  main.c      Engine entry point (CLI)
  model/      Profiles, config, weights, tokenizer, scheduler, forward
  runtime/    Request, sequence builder, generate, decode, preview, refiner
  cuda/       Hand-written CUDA kernels (norm, rope, gemm, attn, act, embed,
              residual, sched, support)
  io/         JSON, safetensors, GGUF, sha256 readers
  image/      PNG encode/decode
tests/        C test harnesses (unit + integration)
tools/        Python oracle helpers (freeze, capture, guard/ledger, gguf convert)
config/       Versioned locks and model profiles (oracle.lock, models.lock,
              dev.json, base.json, manifests)
scripts/      Bootstrap scripts (checkout oracle, download models, freeze)
docs/         Milestone specs, status, contracts, closeout evidence
models/       Git-ignored model weights (dev/, base/)
python/       Git-ignored frozen oracle checkout (validation only)
build/        Git-ignored build output (build/hidream, test binaries)
artifacts/    Git-ignored runtime evidence (golden fixtures, run ledger)
```

## Model profiles

| Profile | steps | guidance | shift | scheduler | dtype |
|---------|-------|----------|-------|-----------|-------|
| **dev** | 28 | 0.0 | 1.0 | flash | BF16 |
| **base** | 50 | 5.0 | 3.0 | default (UniPC) | BF16 |

Frozen model dims: H=4096, NH=32, NKV=8, HD=128, ff_hidden=12288,
head_out=3072, NLAYERS=36, PATCH=32. Weights live in `models/dev` and
`models/base` (8 safetensors shards each), pinned to immutable revisions in
`config/models.lock`. The frozen oracle is pinned by commit SHA in
`config/oracle.lock`.

## Build & run

Requires a CUDA 13 toolchain, cuBLAS (`libcublas.so.13` / `libcublasLt.so.13`)
and cuDNN (`libcudnn.so`, cuDNN 9.20) with the cuDNN C++ Frontend vendored in
`third_party/cudnn-frontend/`; target is NVIDIA GB10, `-arch=sm_121`.

```sh
make            # build the engine (build/hidream)
make test       # build all test binaries
make clean
```

Timing instrumentation (`-DO1_DEBUG_TIMING`) and a `make timing` target are
available for the M2 pre-baseline. The production GEMM backend is persistent
cuBLAS (`cublasGemmEx`, BF16 in/out, FP32 accumulate); the hand-written
reference GEMM remains selectable via `hd_gemm_set_backend(0)` for
correctness/debug only.

## GGUF weight pack (M3 loader)

The engine can load weights from either the raw safetensors shards or a
single materialized **GGUF v3 pack**. The pack is the recommended production
format: it is already BF16, 256-byte aligned, and in production tensor order,
so runtime loading is one sequential `file -> pinned staging -> CUDA arena`
stream (no JSON, no per-tensor lookup, no cast, no per-tensor `cudaMalloc`).

Convert the safetensors shards once (offline, needs numpy):

```sh
python3 tools/hidream_convert.py \
    --source models/dev \
    --output artifacts/models/hidream-o1-dev-bf16.gguf \
    --profile dev \
    --revision b6acc2fe452b3120430620dc4354fa442ee081ea
```

Then point the engine at the pack with `--model-dir`:

```sh
./build/hidream --model dev --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
    --prompt "a teapot" --steps 28 --seed 123456 --output out.png
```

Measured on GB10 (Dev 2048², 28 steps, BF16):

| Loader | FILE_READ | MODEL_LOAD | disk GB/s |
|--------|-----------|------------|-----------|
| safetensors legacy (per-tensor) | 29.1 s | 30.1 s | 1.21 |
| safetensors pipelined (arena + async) | 17.5 s | 26.9 s | 2.01 |
| **GGUF pack** | **1.9 s** | **4.7 s** | **9.3** |

Output is bit-identical across all three loaders (verified 28-step,
cos = 1.0). The GGUF reader (`src/io/gguf.c`) is a minimal C parser with no
ggml/llama.cpp dependency; `test-gguf` validates header parse and payload
round-trip against the source safetensors.

## CLI usage

```
build/hidream [options]
  --model dev|base        profile to use (default: dev)
  --prompt TEXT           user prompt
  --mode t2i|edit|personalize|...   generation mode (default: t2i)
  --ref-image PATH        reference image (repeatable, max 20)
  --ref-image NAME=PATH   named reference; refer to it as @NAME in --prompt
  --verbose               print the reference alias mapping + expanded prompt
  --no-progress           disable the generation progress bar
  --keep-original-aspect  single ref: derive output dims from ref
  --layout-bboxes JSON    layout bboxes for personalize+layout
  --width N               output width (default: 2048)
  --height N              output height (default: 2048)
  --steps N               inference steps (default per profile)
  --seed N                RNG seed (default: 123456)
  --scheduler flash|default|flow_match   (default per profile)
  --guidance F            CFG scale (default per profile)
  --shift F               scheduler shift (default per profile)
  --noise-start F         noise_scale_start (default 8.0)
  --noise-end F           noise_scale_end (default 8.0)
  --noise-clip F          noise_clip_std (default 8.0)
  --lora FILE[:MULT]      apply LoRA adapter (repeatable)
  --output PATH           output PNG path (default: output.png)
  --model-dir DIR         override profile local_path; a path ending in
                          .gguf loads the materialized GGUF weight pack
  --device N              CUDA device index (default: 0)
```

Example:

```sh
./build/hidream --model dev --prompt "a teapot" --steps 28 --seed 123456 \
  --output out.png
```

## Named reference images

References can carry an optional semantic **name** with `--ref-image NAME=PATH`,
and the prompt can refer to them with `@NAME`:

```sh
./build/hidream --model dev --mode personalize \
  --ref-image person=person.jpg \
  --ref-image shirt=shirt.jpg \
  --prompt '@person wearing @shirt' \
  --steps 28 --seed 123456 --output out.png
```

Every reference also gets an automatic alias `@refN` (N = 1-based position on
the command line), so `@ref1`, `@ref2`, … always work, with or without an
explicit name:

```sh
./build/hidream --model dev --mode personalize \
  --ref-image face.jpg --ref-image clothes.jpg \
  --prompt 'Use the identity from @ref1 and the clothes from @ref2' \
  --steps 28 --seed 123456 --output out.png
```

Key points:

- `@name` is a **prompt-side alias only**. It does **not** add a new model
  token, change the tokenizer vocabulary, or modify the reference image
  encoding. Aliases are expanded away before tokenization.
- References retain their **original command-line order**; aliases never
  reorder the reference tensors, `ref_patches`, `image_embeds` or DeepStack.
- A prompt with **no** `@alias` is used byte-identically (the reference
  mapping header is only injected when at least one alias is present).
- Invalid aliases fail closed: duplicates (`duplicate reference alias: x`),
  empty/whitespace names and reserved names starting with `__` are rejected.
  An **unknown** `@token` in the prompt is **not** an error: it is left
  verbatim in the prompt (e.g. `@foo` stays `@foo`).
- Alias grammar: `[A-Za-z_][A-Za-z0-9_-]*` (e.g. `person`, `person_1`,
  `dress-blue`, `pose_front`). `--ref-image a/b=c/d.png` keeps the whole
  string as a path because `a/b` is not a valid alias.
- The mapping is printed with `--verbose` (or `O1_VERBOSE_REF=1`).
- C API: `hd_reference_image.alias` (optional, `NULL` for none) selects the
  same behaviour programmatically.

## Generation progress

When `stderr` is a TTY, generation prints an in-place progress bar for the
denoise loop, with elapsed time and an estimate of the remaining time:

```text
[########------------------]  32% step 9/28  00:12 elapsed  00:25 left
```

- Progress is measured per denoise step, so the fraction is `step/total` and
  the estimate extrapolates linearly from the average step duration.
- The bar is purely frontend: it is driven by `hd_generation_request.progress_cb`.
  It never allocates, synchronizes or touches device state.
- It is disabled automatically when `stderr` is not a TTY (so logs and pipes
  stay clean) and can be turned off explicitly with `--no-progress`.

## Generation examples

The repository includes sample inputs and generated outputs under
[`example_example_assets/`](example_example_assets/). Run the commands below from the
repository root after building `build/hidream`.

| Feature | Mode/options | Input example_assets | Verified output |
|---------|--------------|--------------|-----------------|
| Text-to-image | `--mode t2i` | prompt only | — |
| Instruction-based editing | `--mode edit` | [`edit/test.jpg`](example_example_assets/edit/test.jpg) | [`edit/generated.png`](example_example_assets/edit/generated.png) |
| Multi-reference personalization | `--mode personalize` | [`IP/1.jpg` … `IP/10.jpg`](example_example_assets/IP/) | — |
| Skeleton-guided composition | `--mode personalize` with face, background, pose and part references | [`IP_skeleton/`](example_example_assets/IP_skeleton/) | — |
| Personalization with layout | `--mode layout --layout-bboxes` | [`IP_layout/0.jpg`](example_example_assets/IP_layout/0.jpg), [`IP_layout/1.jpg`](example_example_assets/IP_layout/1.jpg) | — |
| Preserve source aspect ratio | `--keep-original-aspect` | [`edit/test.jpg`](example_example_assets/edit/test.jpg) | — |

The examples use the materialized Dev GGUF at
`artifacts/models/hidream-o1-dev-bf16.gguf`. Omit `--model-dir` to use the
path configured by the selected profile.

> **Use the full Dev schedule for image-quality examples.** `--steps 1` is
> useful only as an execution smoke test; its decoded image may be uniform
> mid-gray and must not be treated as a generated result. The commands below
> therefore use the Dev default of 28 steps.

### Text-to-image (Dev)

```sh
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --prompt "A dog holds a sign that says HiDream-O1-Image release." \
  --width 2048 --height 2048 --steps 28 --seed 42 \
  --output t2i-dev-output.png
```

### Text-to-image (Base + FlowUniPC/CFG)

The Base profile selects the 50-step default FlowUniPC scheduler and CFG:

```sh
./build/hidream --model base \
  --prompt "A cinematic portrait in soft natural light." \
  --width 2048 --height 2048 --steps 50 --seed 42 \
  --output t2i-base-output.png
```

### Instruction-based editing

```sh
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --mode edit \
  --ref-image example_example_assets/edit/test.jpg \
  --prompt "remove the earphones" \
  --width 2048 --height 2048 --steps 28 --seed 123456 \
  --output edit-output.png
```

### Multi-reference personalization

```sh
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --mode personalize \
  --ref-image example_example_assets/IP/1.jpg \
  --ref-image example_example_assets/IP/2.jpg \
  --ref-image example_example_assets/IP/3.jpg \
  --ref-image example_example_assets/IP/4.jpg \
  --ref-image example_example_assets/IP/5.jpg \
  --ref-image example_example_assets/IP/6.jpg \
  --ref-image example_example_assets/IP/7.jpg \
  --ref-image example_example_assets/IP/8.jpg \
  --ref-image example_example_assets/IP/9.jpg \
  --ref-image example_example_assets/IP/10.jpg \
  --prompt "Create a coherent portrait using the supplied subject references." \
  --width 2048 --height 2048 --steps 28 --seed 42 \
  --output personalize-output.png
```

References may also be named and used in the prompt with the
[`@alias` syntax](#named-reference-images).

### Skeleton-guided multi-reference composition

Skeleton conditioning uses the face, background, OpenPose and body-part
images as an ordered multi-reference personalization request:

```sh
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --mode personalize \
  --ref-image example_example_assets/IP_skeleton/0.face.jpg \
  --ref-image example_example_assets/IP_skeleton/0.bg.jpg \
  --ref-image example_example_assets/IP_skeleton/0.openpose.jpg \
  --ref-image example_example_assets/IP_skeleton/0.part_1.jpg \
  --ref-image example_example_assets/IP_skeleton/0.part_2.jpg \
  --ref-image example_example_assets/IP_skeleton/0.part_3.jpg \
  --prompt "Create a realistic try-on image of the person wearing the provided clothing." \
  --width 2048 --height 2048 --steps 28 --seed 42 \
  --output skeleton-output.png
```

### Personalization with layout

Bounding boxes use normalized `[x_min, x_max, y_min, y_max]` coordinates and
follow the same order as the reference images:

```sh
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --mode layout \
  --ref-image person=example_example_assets/IP_layout/0.jpg \
  --ref-image object=example_example_assets/IP_layout/1.jpg \
  --layout-bboxes "[[0.20507812,0.43945312,0.48828125,0.7421875],[0.57617188,0.80078125,0.08789062,0.34179688]]" \
  --prompt "@person and @object arranged according to the supplied layout." \
  --width 2048 --height 2048 --steps 28 --seed 42 \
  --output layout-output.png
```

### Preserve the original aspect ratio

With one reference, `--keep-original-aspect` derives patch-aligned output
dimensions from the source image:

```sh
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --mode edit \
  --ref-image example_example_assets/edit/test.jpg \
  --keep-original-aspect \
  --prompt "remove the earphones" \
  --steps 28 --seed 42 \
  --output edit-keep-aspect-output.png
```

See [`example_example_assets/README.md`](example_example_assets/README.md) for attribution,
additional context and the upstream prompts associated with these example_assets.

## Server / OpenAI-compatible Images API

`build/hidream-server` exposes the same generation engine as the CLI through
an OpenAI-compatible Images API. It keeps **one model resident**: the weights
are loaded once at startup and every request reuses them, so request latency
is the generation itself, not a model reload.

### Startup

```sh
export LD_LIBRARY_PATH=/home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

./build/hidream-server --port 8000 --model dev
```

Options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--host HOST` | `127.0.0.1` | Listen address |
| `--port N` | `8000` | TCP port |
| `--model dev\|base` | `dev` | Profile to load and keep resident |
| `--model-dir PATH` | per-profile GGUF | Weights directory or `.gguf` pack |
| `--api-key SECRET` | none | Require `Authorization: Bearer SECRET` |
| `--cors` | off | Emit permissive CORS headers, answer `OPTIONS` with 204 |
| `--queue-depth N` | `8` | Maximum *waiting* jobs before `429 queue_full` |
| `--max-body-mb N` | `64` | Maximum request body size (`413` beyond it) |
| `--device N` | `0` | CUDA device index |
| `--lora FILE[:MULT]` | none | Merge a LoRA adapter at startup (repeatable) |

At shutdown (`SIGINT`/`SIGTERM`) the server stops accepting, drains the
in-flight request, then prints the preload-lifecycle release gate to stderr:

```text
hidream-server: shutting down
hidream-server: lifecycle open=1 weight_load=1 forward_resolve=1 vision_resolve=1 requests=2 close=1
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Liveness + loaded profile |
| `GET` | `/v1/models` | Advertises the resident model id |
| `POST` | `/v1/images/generations` | JSON text-to-image |
| `POST` | `/v1/images/edits` | `multipart/form-data`, reference images |

### Supported fields

Standard OpenAI fields: `prompt`, `model`, `n` (1–4), `size`, `quality`
(`auto` only), `response_format` (`b64_json` only), `output_format` (`png`
only), `background` (`auto`/`opaque`), `user`, and — on edits — `image` /
`image[]` plus `mask` (rejected).

Engine-specific fields (prefixed `o1_`):

| Field | Meaning |
|-------|---------|
| `o1_mode` | `t2i`, `edit`, `personalize`/`multi-ref`, `personalize_layout`/`layout` |
| `o1_seed` | 64-bit seed; with `n>1` image *i* uses `seed+i` |
| `o1_steps` | Denoising steps (1–64) |
| `o1_guidance_scale`, `o1_shift` | Sampler overrides |
| `o1_scheduler` | `flash`, `flow_match`, `default`/`unipc` |
| `o1_noise_start`, `o1_noise_end`, `o1_noise_clip` | Noise schedule overrides |
| `o1_keep_original_aspect` | Derive output dims from a single reference |
| `o1_exact_size` | Fail instead of snapping to a supported bucket |
| `o1_reference_aliases` | JSON array, one `@name` per input image |
| `o1_layout_bboxes` | Layout description for `personalize_layout` |
| `o1_verbose` | Log the reference mapping and expanded prompt to stderr |

### Known unsupported fields

`mask` (`mask_not_supported`), `background=transparent`
(`unsupported_background`), any `output_format` other than `png`
(`unsupported_output_format`), `response_format=url`, `compression`,
`quality` presets, and per-request `o1_lora` (adapters are startup-global).
`o1_mode=personalize_skeleton` is rejected: skeleton conditioning is not
implemented by the server.

### Resolution snapping

The engine only generates at frozen buckets (`2048x2048`, `2304x1728`,
`1728x2304`, `2560x1440`, …). A requested `size` is snapped to the nearest
bucket and the server logs the substitution:

```text
hidream-server: request size 1024x1024 snapped to 2048x2048
```

`1024x1024` is **not** a bucket. Pass `o1_exact_size: true` to fail with
`400 unsupported_size` instead of snapping.

### curl examples

```sh
# Text to image
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"hidream-o1-image-dev","prompt":"a red apple on a wooden table",
       "size":"2048x2048","o1_seed":42,"o1_steps":28}' \
  | python3 -c 'import sys,json,base64;open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'

# Edit (single reference)
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F model=hidream-o1-image-dev \
  -F prompt='remove the earphones' \
  -F o1_mode=edit -F o1_seed=7 \
  -F image=@example_example_assets/edit/test.jpg \
  -o edit.json

# Multi-reference personalization with named references
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F model=hidream-o1-image-dev \
  -F prompt='a photo of @subject in the style of @style' \
  -F o1_mode=personalize \
  -F 'o1_reference_aliases=["subject","style"]' \
  -F image=@example_example_assets/IP_2.jpg \
  -F image=@example_example_assets/edit/test.jpg \
  -o multiref.json
```

An `@name` that has no matching entry in `o1_reference_aliases` is left
**verbatim** in the prompt and does not fail the request.

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

r = client.images.generate(
    model="hidream-o1-image-dev",
    prompt="A cinematic photograph of a red fox in snow",
    size="2048x2048",
    extra_body={"o1_seed": 42, "o1_steps": 28},
)
open("fox.png", "wb").write(__import__("base64").b64decode(r.data[0].b64_json))
```

A full end-to-end SDK smoke test (generate + single-reference edit +
multi-reference personalization) lives in
[`tests/server/openai_client_smoke.py`](tests/server/openai_client_smoke.py);
the failure-path suite lives in
[`tests/server/failure_cases.py`](tests/server/failure_cases.py).

