---
license: mit
language:
  - en
library_name: o1.c
pipeline_tag: text-to-image
tags:
  - text-to-image
  - image-editing
  - personalization
  - gguf
  - hidream
  - o1.c
---

# HiDream-O1-Image — GGUF conversions for the native o1.c runtime

This repository hosts **GGUF conversions** of the HiDream-O1-Image model
family, packaged for the native **o1.c** inference engine (C/CUDA, no
PyTorch required at runtime).

> **These are GGUF conversions for the native o1.c runtime.**
>
> **Original model authors: [HiDream-ai](https://huggingface.co/HiDream-ai).**
> All model weights, architecture and training originate from the upstream
> HiDream-ai repositories. This repository does **not** replace the upstream
> model repository; it only redistributes a reformatted (GGUF) copy of the
> released weights under the same MIT license.
>
> If you want the original PyTorch/safetensors checkpoints, or the
> authoritative model card, training details and citation, please use the
> upstream repositories listed under *Provenance*.

---

## Available artifacts

All artifacts are **BF16** GGUF files with the o1.c layout version 1.

| File | Profile | Variant | Steps | Size | SHA256 |
|------|---------|---------|-------|------|--------|
| `hidream-o1-dev-2604-bf16.gguf` | `dev` | `dev-2604` | 28 | 17 609 841 152 B | `7c405035a9225b721ba19b1a1f4e7f1cb937ddea7dbb7207e5ad677c3e0a7f5d` |
| `hidream-o1-dev-bf16.gguf`      | `dev` | `dev`      | 28 | 17 609 841 152 B | `4b1141733ec5a99fcbc2994bf720e8ef6ade31ec3e9378e471e73f3af6ce8fae` |
| `hidream-o1-base-bf16.gguf`     | `base`| `base`     | 50 | 17 609 841 152 B | `07f5efb347cfd9264c8dadd3848bb8c6053633b37a05b0680d920a6409de011b` |

`hidream-o1-dev-2604-bf16.gguf` is the recommended artifact.

The checksums are also published in [`SHA256SUMS`](./SHA256SUMS) and in the
o1.c engine repository's `models/manifest.json`.

**Q4 is not available.** The o1.c runtime currently accepts only
F32/F16/BF16 GGUF tensors; there is no quantized weight path yet. Do not
expect or request Q4 files from this repository until a separate
quantization gate is completed upstream in o1.c.

---

## Provenance

Each artifact is a byte-exact re-layout of one immutable upstream revision.
The revision is a commit SHA, never a branch name.

| Artifact | Source repository | Source revision | License |
|----------|-------------------|-----------------|---------|
| `hidream-o1-dev-2604-bf16.gguf` | [`HiDream-ai/HiDream-O1-Image-Dev-2604`](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604) | `b6acc2fe452b3120430620dc4354fa442ee081ea` | MIT |
| `hidream-o1-dev-bf16.gguf`      | [`HiDream-ai/HiDream-O1-Image-Dev`](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev) | `c0bada0e15c54a9f96a6d1ecc35575b32bc21544` | MIT |
| `hidream-o1-base-bf16.gguf`     | [`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image) | `0b0901d99f200389e138c61946af1185f5f49a13` | MIT |

Upstream license: **MIT** (declared in the model metadata of all three source
repositories). The upstream repositories do not ship a standalone `LICENSE`
file, so the MIT text is reproduced in [`LICENSE`](./LICENSE) for
convenience and no additional terms are imposed by this repository.

No weights were modified, fine-tuned, merged, pruned or re-quantized. The
conversion is a layout/dtype-preserving re-serialization: tensor values are
the upstream values in the upstream dtype (BF16), stored in the o1.c GGUF
container with its metadata keys added.

---

## GGUF metadata

Every artifact carries at least the following keys, which the o1.c runtime
reads to configure itself (so that a single `--model-path` is enough):

```text
general.architecture        = hidream_o1
general.name                = <model name>
hidream.profile             = dev | base
hidream.variant             = dev | dev-2604 | base
hidream.revision            = <upstream commit sha>
hidream.dtype               = bf16
hidream.quantization        = bf16
hidream.num_layers          = <int>
hidream.layout_version      = 1
```

Execution profile derived from `hidream.profile`:

* `dev`  → 28 steps, guidance 0, shift 1
* `base` → 50 steps, guidance 5, shift 3

---

## How to use

Install the o1.c engine, then download and run:

```bash
# from an o1.c checkout
./scripts/download_model.sh dev-2604

./build/hidream --model-path models/hidream-o1-dev-2604-bf16.gguf \
    --prompt "a red fox in a snowy forest, golden hour" \
    --width 2048 --height 2048 --seed 42 --output fox.png
```

You can also download these files directly and pass the path to
`--model-path`. See the o1.c repository for installation and the full CLI
reference.

---

## Acknowledgements

Model architecture, training and original weights: **HiDream-ai**.
GGUF conversion and the o1.c runtime: the o1.c project.

If you use these weights, please cite the original HiDream-O1-Image work as
described in the upstream model cards.
