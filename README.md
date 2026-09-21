# o1.c

A native **C/CUDA inference engine** for the
[HiDream-O1-Image](https://huggingface.co/HiDream-ai) foundation image model.

No Python at runtime. No PyTorch, no `transformers`, no diffusers — just a
single C binary that loads a GGUF model and generates images on your GPU.

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --prompt "a red fox in a snowy forest, golden hour" \
    --width 2048 --height 2048 --seed 42 --output fox.png
```

---

## 1. What is o1.c

o1.c re-implements the HiDream-O1-Image transformer in hand-written CUDA
kernels and runs it directly from a GGUF weight file. It is aimed at NVIDIA
GPUs and is validated on the GB10 (DGX Spark).

It supports **text-to-image**, **instruction-based image editing**,
**multi-reference personalization**, **layout/skeleton conditioning** and
**LoRA adapters**, all from the same binary, plus an optional
OpenAI-compatible HTTP server.

The engine is deterministic: the same prompt, seed and model produce the same
image every time.

## 2. Features

- **Text-to-image** at up to 2048×2048 (and other supported buckets).
- **Image editing** from a single reference image, with an instruction prompt.
- **Multi-reference personalization** with up to 20 references.
- **Named references** (`--ref-image NAME=PATH` + `@NAME` in the prompt).
- **Layout conditioning** with normalized bounding boxes.
- **Skeleton-guided composition** via ordered multi-reference inputs.
- **LoRA** adapters merged at load time.
- **OpenAI-compatible server** exposing `/v1/images/generations` and
  `/v1/images/edits`.
- **GGUF** weights, 256-byte aligned, loaded in one sequential stream.
- **Deterministic output** for a fixed prompt/seed/model.

## 3. Requirements / supported hardware

| Requirement | Minimum |
|-------------|---------|
| GPU | NVIDIA GPU; validated on GB10 (DGX Spark), compute capability 12.1 |
| Driver | a driver able to run the installed CUDA toolkit (580.x validated) |
| CUDA toolkit | 13.0 validated (`nvcc`, headers, `libcudart`) |
| cuBLAS / cuBLASLt | shipped with the CUDA toolkit |
| cuDNN | 9.x (9.20 validated) |
| Compiler | C11 + C++17 host compiler (gcc 13.3 validated) |
| `make` | GNU make |
| Disk | ≈ 18 GB per BF16 model |
| VRAM | ≈ 0.9 GB (T2I 2048) to ≈ 5.2 GB (Base 2048) reported RSS |

Run `./scripts/setup.sh` to check all of these at once. See
[Installation](#5-installation).

## 4. Quick start

```sh
git clone https://github.com/luca-saggese/o1.c.git
cd o1.c

./scripts/setup.sh                 # check your environment
source scripts/env.sh              # resolve CUDA / cuDNN paths
make                               # build build/hidream

./scripts/download_model.sh dev-2604

./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --prompt "a red fox in a snowy forest, golden hour" \
    --width 2048 --height 2048 --seed 42 --output fox.png
```

## 5. Installation

### 5.1 Check the environment

```sh
./scripts/setup.sh
```

This validates the GPU, compute capability, driver, `nvcc`, CUDA headers,
`libcudart`, cuBLAS, cuBLASLt, cuDNN, the compiler and `make`. It only
reports; it never installs anything. If something is missing it prints exactly
what to do.

### 5.2 Resolve library paths

```sh
source scripts/env.sh
```

This resolves `CUDA_HOME` and `CUDNN_HOME` from standard locations (or honours
existing overrides) and exports `PATH` and `LD_LIBRARY_PATH`. It is safe to
source repeatedly.

If cuDNN lives somewhere unusual, point at it explicitly:

```sh
export CUDNN_HOME=/path/to/cudnn
source scripts/env.sh
```

cuDNN 9.x can be installed without root via
`pip install nvidia-cudnn-cu13`, or from the NVIDIA native tarball into
`/usr/local/cudnn`. Both are detected automatically. See
[`docs/SETUP.md`](docs/SETUP.md) for details.

### 5.3 Build

```sh
make            # build/hidream
make server     # build/hidream-server (optional)
```

## 6. Download a model

```sh
./scripts/download_model.sh dev-2604     # recommended
./scripts/download_model.sh dev
./scripts/download_model.sh base
```

The script reads [`models/manifest.json`](models/manifest.json), downloads
from Hugging Face, supports resume, shows progress, **verifies SHA256** and
fails on a checksum mismatch. It prints the exact `--model-path` command to
use when it finishes.

Useful options:

```sh
./scripts/download_model.sh list                 # show available models
./scripts/download_model.sh dev-2604 --dir /data # choose destination
./scripts/download_model.sh dev-2604 --force     # re-download
```

Available models:

| id | Model | Steps | Size |
|----|-------|-------|------|
| `dev-2604` | HiDream-O1-Image-Dev-2604 | 28 | 17.6 GB |
| `dev` | HiDream-O1-Image-Dev | 28 | 17.6 GB |
| `base` | HiDream-O1-Image (Base/Full) | 50 | 17.6 GB |

The artifacts are published at
[`saggeseluca/o1.c-models`](https://huggingface.co/saggeseluca/o1.c-models)
on Hugging Face. The download script reads the URLs and checksums from
[`models/manifest.json`](models/manifest.json) — no URL is hard-coded in the
script, so you never need to know the repository path by hand.

**Q4 is not available.** The runtime currently supports only F32/F16/BF16
GGUF tensors, so there is no quantized release yet. Do not expect Q4 files.

## 7. Generate an image

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --prompt "A serene mountain lake at sunrise, photorealistic" \
    --width 2048 --height 2048 --steps 28 --seed 42 \
    --output lake.png
```

Use the model's default step count unless you have a reason to change it
(28 for Dev, 50 for Base). For text-to-image, `--steps 1` is a useful execution
smoke test, but its image is not a real generation. Edit and personalization
modes need a full schedule: very low step counts produce out-of-range latents
and are rejected rather than written out.

## 8. Edit an image

Pass a reference image and an instruction prompt:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --mode edit \
    --ref-image example_assets/edit/test.jpg \
    --prompt "remove the earphones" \
    --width 2048 --height 2048 --steps 28 --seed 42 \
    --output edit.png
```

To keep the source aspect ratio instead of forcing the requested size:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --mode edit \
    --ref-image example_assets/edit/test.jpg \
    --keep-original-aspect \
    --prompt "remove the earphones" \
    --steps 28 --seed 42 \
    --output edit-aspect.png
```

## 9. Multi-reference personalization

Pass several references; they are combined into one coherent subject:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --mode personalize \
    --ref-image example_assets/IP/1.jpg \
    --ref-image example_assets/IP/2.jpg \
    --ref-image example_assets/IP/3.jpg \
    --prompt "Create a coherent portrait using the supplied subject references." \
    --width 2048 --height 2048 --steps 28 --seed 42 \
    --output personalize.png
```

References can be **named** and referred to in the prompt with `@name`:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --mode personalize \
    --ref-image person=person.jpg \
    --ref-image shirt=shirt.jpg \
    --prompt '@person wearing @shirt' \
    --steps 28 --seed 42 --output outfit.png
```

Every reference also gets an automatic alias `@ref1`, `@ref2`, … by position.
An unknown `@token` is left verbatim in the prompt rather than failing. Use
`--verbose` to print the alias mapping.

## 10. Layout / skeleton conditioning

**Layout.** Provide normalized `[x_min, x_max, y_min, y_max]` boxes, one per
reference, in the same order:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --mode layout \
    --ref-image person=example_assets/IP_layout/0.jpg \
    --ref-image object=example_assets/IP_layout/1.jpg \
    --layout-bboxes "[[0.205,0.439,0.488,0.742],[0.576,0.801,0.088,0.342]]" \
    --prompt "@person and @object arranged according to the supplied layout." \
    --width 2048 --height 2048 --steps 28 --seed 42 \
    --output layout.png
```

**Skeleton.** Skeleton conditioning is ordered multi-reference personalization
using face, background, pose and body-part references:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --mode personalize \
    --ref-image example_assets/IP_skeleton/0.face.jpg \
    --ref-image example_assets/IP_skeleton/0.bg.jpg \
    --ref-image example_assets/IP_skeleton/0.openpose.jpg \
    --ref-image example_assets/IP_skeleton/0.part_1.jpg \
    --ref-image example_assets/IP_skeleton/0.part_2.jpg \
    --ref-image example_assets/IP_skeleton/0.part_3.jpg \
    --prompt "Create a realistic try-on image of the person wearing the provided clothing." \
    --width 2048 --height 2048 --steps 28 --seed 42 \
    --output skeleton.png
```

## 11. Model variants

| Variant | GGUF | Profile | Default steps | Guidance | Shift | Scheduler |
|---------|------|---------|---------------|----------|-------|-----------|
| Dev-2604 | `hidream-o1-dev-2604-bf16.gguf` | `dev` | 28 | 0 | 1 | flash |
| Dev | `hidream-o1-dev-bf16.gguf` | `dev` | 28 | 0 | 1 | flash |
| Base/Full | `hidream-o1-base-bf16.gguf` | `base` | 50 | 5 | 3 | default (UniPC) |

Dev-2604 and Dev share the same execution profile but are distinct
checkpoints. Base requires its full 50-step schedule; a short recipe will not
produce a valid image.

The engine reads the execution profile from the GGUF metadata, so
`--model-path` alone is enough — you never select a profile by hand.

## 12. CLI reference

```
build/hidream [options]

  --model-path PATH       production GGUF model file (profile inferred
                          from GGUF metadata; this is the public interface)
  --prompt TEXT           user prompt
  --mode t2i|edit|personalize|layout|...   generation mode (default: t2i)
  --ref-image PATH        reference image (repeatable, max 20)
  --ref-image NAME=PATH   named reference; use @NAME in --prompt
  --keep-original-aspect  single ref: derive output dims from ref
  --layout-bboxes JSON    layout bboxes for personalize+layout
  --verbose               print reference alias mapping / expanded prompt
  --no-progress           disable the generation progress bar
  --width N               output width (default: 1024)
  --height N              output height (default: 1024)
  --steps N               inference steps (default per model profile)
  --seed N                RNG seed (default: 123456)
  --scheduler flash|default|flow_match   (default per model profile)
  --guidance F            CFG scale (default per model profile)
  --shift F               scheduler shift (default per model profile)
  --output PATH           output PNG path (default: output.png)
  --device N              CUDA device index (default: 0)
  --noise-start F         noise_scale_start (default: 8.0)
  --noise-end F           noise_scale_end (default: 8.0)
  --noise-clip F          noise_clip_std (default: 8.0)
  --lora FILE[:MULT]      apply LoRA adapter (repeatable)
  -h, --help              show this help
```

Sizes are snapped to supported buckets; `1024x1024` is not a bucket. Requests
are aligned to the model's patch grid.

## 13. LoRA

Apply one or more adapters, optionally scaled:

```sh
./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --lora my-style.safetensors:0.8 \
    --prompt "a portrait in my style" \
    --steps 28 --seed 42 --output lora.png
```

Adapters are merged into the weights at load time (`W' = W + mult · (α/r) ·
up·down`), so there is no runtime cost. Only linear targets are supported;
unsupported adapters fail explicitly rather than being partially applied.
See [`docs/LORA_IMPLEMENTATION.md`](docs/LORA_IMPLEMENTATION.md).

## 14. Server / OpenAI-compatible API

`build/hidream-server` keeps **one model resident** and serves the same engine
over an OpenAI-compatible Images API.

```sh
source scripts/env.sh

./build/hidream-server --port 8000 \
    --model-path models/hidream-o1-dev-2604-bf16.gguf
```

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
  -F image=@example_assets/edit/test.jpg \
  -o edit.json
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--host HOST` | `127.0.0.1` | Listen address |
| `--port N` | `8000` | TCP port |
| `--model-path PATH` | — | Production GGUF model file |
| `--api-key SECRET` | none | Require an `Authorization` header |
| `--cors` | off | Permissive CORS headers |
| `--queue-depth N` | `8` | Max waiting jobs before `429` |
| `--max-body-mb N` | `64` | Max request body size |
| `--device N` | `0` | CUDA device index |
| `--lora FILE[:MULT]` | none | Merge a LoRA adapter at startup |

Engine-specific request fields are prefixed `o1_` (`o1_seed`, `o1_steps`,
`o1_mode`, `o1_guidance_scale`, `o1_shift`, `o1_scheduler`,
`o1_reference_aliases`, `o1_layout_bboxes`, …). `response_format` must be
`b64_json` and `output_format` must be `png`.

## 15. Performance expectations

On a GB10 (DGX Spark), with the model loaded:

| Workload | Recipe | Wall clock |
|----------|--------|-----------|
| Text-to-image | 2048×2048, 28 steps | ≈ 86 s |
| Edit (1 reference) | 2048×2048, 28 steps | ≈ 196 s |
| Text-to-image (Base) | 2048×2048, 50 steps, CFG | ≈ 320 s |

Model loading is separate (a few seconds from a GGUF) and happens once per
process. The server loads once and reuses the resident model for every
request. Edit is slower than text-to-image because the reference's pixel
tokens are appended to the denoiser sequence at every step.

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for the full breakdown.

## 16. Troubleshooting

**`error while loading shared libraries: libcudnn.so.9`**
cuDNN is not on the loader path. Run `source scripts/env.sh`, or export
`LD_LIBRARY_PATH="$CUDNN_HOME/lib:$LD_LIBRARY_PATH"`. Confirm with
`ldd build/hidream | grep cudnn`.

**`cudnn.h: No such file or directory` / cuDNN not found at build time**
cuDNN is not installed or not detected. Run `./scripts/setup.sh`, install
cuDNN 9.x (`pip install nvidia-cudnn-cu13`), or set
`CUDNN_HOME=/path/to/cudnn`.

**`nrmse` / `test_full_forward` failure from `make test`**
This is a known, pre-existing numerical exception in the test suite. It does
not affect production generation. Run individual test targets if you need to
bypass it.

**Generation is very slow the first time**
The first run includes CUDA module loading and cuDNN plan construction.
Subsequent runs in the same process are faster.

**`final latent range … out of bounds` with Base**
Base requires its full 50-step schedule with CFG. Do not shorten it.

**Output size differs from `--width`/`--height`**
Sizes snap to supported buckets (multiples of the patch grid). `1024x1024` is
not a bucket.

## 17. Building from source

```sh
./scripts/setup.sh
source scripts/env.sh
make            # build/hidream
make server     # build/hidream-server
make test       # build the native test suite
make clean
```

`CUDNN_HOME` and `CUDA_HOME` are auto-detected; override them if needed:

```sh
make CUDNN_HOME=/opt/cudnn
```

The build targets `-arch=sm_121` by default. Engineering details, test
targets and the internal safetensors development path are documented in
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/SETUP.md`](docs/SETUP.md).

## 18. Model provenance / acknowledgements

The model weights are the work of **[HiDream-ai](https://huggingface.co/HiDream-ai)**.
o1.c only converts them to GGUF and provides a native runtime. This project
does not replace the upstream model repository — please use it for the
authoritative model cards and citation.

| Release GGUF | Source repository | Revision | License |
|--------------|-------------------|----------|---------|
| `hidream-o1-dev-2604-bf16.gguf` | `HiDream-ai/HiDream-O1-Image-Dev-2604` | `b6acc2fe…` | MIT |
| `hidream-o1-dev-bf16.gguf` | `HiDream-ai/HiDream-O1-Image-Dev` | `c0bada0e…` | MIT |
| `hidream-o1-base-bf16.gguf` | `HiDream-ai/HiDream-O1-Image` | `0b0901d9…` | MIT |

The GGUF conversions are redistributed from
[`saggeseluca/o1.c-models`](https://huggingface.co/saggeseluca/o1.c-models),
which is **not** a replacement for the upstream repositories. See
[`docs/RELEASE_MODELS.md`](docs/RELEASE_MODELS.md) for full provenance and
reproduction commands.

## 19. License

The engine source is provided as-is for use with the HiDream-O1-Image models.
Model weights remain subject to the **upstream license (MIT)** and are not
redistributed in this repository; use `scripts/download_model.sh` or the
upstream Hugging Face repositories.
