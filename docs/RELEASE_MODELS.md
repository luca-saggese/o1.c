# Release Model Artifacts — R2

Reproducible BF16 GGUF conversion definitions and validation results for the
three release model families.

Conversion is defined **only** in [`release/models.def.json`](../release/models.def.json)
and executed by [`scripts/build_release_models.sh`](../scripts/build_release_models.sh).
Nothing else may define these artifacts.

```bash
./scripts/build_release_models.sh              # all three
./scripts/build_release_models.sh dev-2604     # one
./scripts/build_release_models.sh --no-download  # use existing checkouts
```

The script downloads each pinned revision into a scratch directory, verifies the
checkout against upstream before converting, then writes
`release/models/SHA256SUMS` and `release/models/models.built.json`.

## 1. Release families

| Release family | `id` | GGUF file | `hidream.profile` | `hidream.variant` |
|---|---|---|---|---|
| HiDream-O1-Image-Dev-2604 | `dev-2604` | `hidream-o1-dev-2604-bf16.gguf` | `dev` | `dev-2604` |
| HiDream-O1-Image-Dev | `dev` | `hidream-o1-dev-bf16.gguf` | `dev` | `dev` |
| HiDream-O1-Image (Base/Full) | `base` | `hidream-o1-base-bf16.gguf` | `base` | `base` |

Dev and Dev-2604 share the same *execution profile* (`dev`: 28 steps, guidance 0,
shift 1, Flash scheduler) while retaining distinct provenance metadata
(`general.name`, `hidream.variant`, `hidream.revision`).

## 2. Provenance

| `id` | Source HF repository | Immutable revision |
|---|---|---|
| `dev-2604` | `HiDream-ai/HiDream-O1-Image-Dev-2604` | `b6acc2fe452b3120430620dc4354fa442ee081ea` |
| `dev` | `HiDream-ai/HiDream-O1-Image-Dev` | `c0bada0e15c54a9f96a6d1ecc35575b32bc21544` |
| `base` | `HiDream-ai/HiDream-O1-Image` | `0b0901d99f200389e138c61946af1185f5f49a13` |

All three upstream repositories publish the same
`model.safetensors.index.json` (sha256 `33865f9d84679a85…`), so the index alone
does **not** identify a checkpoint. The script therefore verifies the checkout
against the pinned revision through the Hub API, and the artifacts below were
confirmed to differ in their shard payloads (for example
`model-00001-of-00008.safetensors`: `f34477502c47` dev-2604, `575a1b54a028`
dev, `db5d56d92c14` base).

## 3. Artifacts

Engine commit at conversion time: `f113487` (R1).
Converter: `tools/hidream_convert.py`, GGUF v3, `general.alignment = 256`,
all tensors BF16.

| `id` | File | Size (bytes) | SHA256 |
|---|---|---|---|
| `dev-2604` | `hidream-o1-dev-2604-bf16.gguf` | 17 609 841 152 | `7c405035a9225b721ba19b1a1f4e7f1cb937ddea7dbb7207e5ad677c3e0a7f5d` |
| `dev` | `hidream-o1-dev-bf16.gguf` | 17 609 841 152 | `4b1141733ec5a99fcbc2994bf720e8ef6ade31ec3e9378e471e73f3af6ce8fae` |
| `base` | `hidream-o1-base-bf16.gguf` | 17 609 841 152 | `07f5efb347cfd9264c8dadd3848bb8c6053633b37a05b0680d920a6409de011b` |

All three carry 759 tensors and a 17 609 778 176-byte payload; the files differ
only in payload content and metadata.

### Exact conversion commands

```bash
python3 tools/hidream_convert.py \
    --source models/dev \
    --output release/models/hidream-o1-dev-2604-bf16.gguf \
    --profile dev --variant dev-2604 \
    --revision b6acc2fe452b3120430620dc4354fa442ee081ea

python3 tools/hidream_convert.py \
    --source models/dev-orig \
    --output release/models/hidream-o1-dev-bf16.gguf \
    --profile dev --variant dev \
    --revision c0bada0e15c54a9f96a6d1ecc35575b32bc21544

python3 tools/hidream_convert.py \
    --source models/base \
    --output release/models/hidream-o1-base-bf16.gguf \
    --profile base --variant base \
    --revision 0b0901d99f200389e138c61946af1185f5f49a13
```

Reproducibility was confirmed by running the script twice: all three SHA256
values were identical across runs.

## 4. Metadata validation

Every artifact parses through the engine's GGUF reader:

```
$ ./build/test_gguf release/models/hidream-o1-dev-2604-bf16.gguf
GGUF: 759 tensors, alignment=256, payload=17609778176 bytes
  arch=hidream_o1 profile=dev revision=b6acc2fe452b3120430620dc4354fa442ee081ea dtype=bf16 layers=36
  name=HiDream-O1-Image-dev-2604 variant=dev-2604 quantization=bf16 layout_version=1
GGUF TEST: PASS
```

| `id` | arch | profile | variant | quantization | layers | layout |
|---|---|---|---|---|---|---|
| `dev-2604` | `hidream_o1` | `dev` | `dev-2604` | `bf16` | 36 | 1 |
| `dev` | `hidream_o1` | `dev` | `dev` | `bf16` | 36 | 1 |
| `base` | `hidream_o1` | `base` | `base` | `bf16` | 36 | 1 |

All three pass `test_gguf` (header, metadata, tensor-table integrity and
payload round-trip against the source safetensors).

## 5. Validation results

| `id` | Recipe | Result | Output |
|---|---|---|---|
| `dev-2604` | 2048², 28 steps, seed 42, Flash | PASS, 1 m 23 s | sha256 `4e79cd4ab639aa17…` — **bit-identical to the R0 frozen baseline** |
| `dev` | 2048², 28 steps, seed 42, Flash | PASS | sha256 `315b115c0ce42f56…` (distinct checkpoint, distinct image) |
| `base` | 2048², **50 steps**, seed 42, guidance 5, shift 3, default scheduler | PASS, 4 m 36 s, 920 MB RSS | sha256 `46a9a1453c3815e3…` — **bit-identical to the R0 frozen baseline** |

Smoke generations (512×512, 4 steps) succeed for `dev` and `dev-2604`.
Base does **not** support a 4-step recipe: its 50-step FlowUniPC schedule with
CFG is what keeps the latent in range, and a truncated run ends with
`final latent range … out of bounds`. This is expected behaviour of the Base
recipe, not a defect of the artifact.

All three outputs were confirmed to be real, non-degenerate images
(2048×2048, 8-bit RGB, mean ≈ 105–110, std ≈ 62–73, full 0–255 range).

## 6. Gate result

```
R2 three BF16 release GGUFs: PASS
```

The artifacts are built and validated but **not uploaded** — publishing is gate
R4.
