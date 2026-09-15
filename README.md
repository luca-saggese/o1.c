# HiDream O1 Image Inference Engine (`o1.c`)

A from-scratch **native C/CUDA inference engine** for the
[HiDream O1 Image](https://huggingface.co/HiDream-ai) foundation image model,
targeted at the **NVIDIA GB10** (DGX Spark) platform. The engine re-implements
the model's transformer math in hand-written CUDA kernels, validated
tensor-by-tensor against a frozen Python oracle, so that **production inference
never depends on Python**.

> **Status:** M0 (oracle/model/startup pinning) and M1.1 (weight ingestion into
> deterministic CUDA buffers) are complete. M1.2 (reference CUDA transformer
> primitives) is in flight. See [`MILESTONES.md`](MILESTONES.md) for the
> authoritative milestone ladder and [`docs/M1_STATUS.md`](docs/M1_STATUS.md)
> for current execution state.

---

## What this project is

`o1.c` builds a deterministic, reproducible inference path for HiDream O1
Image using only C code and hand-written CUDA kernels. The pipeline is

```
GGUF/config lock  ->  weight ingestion (safetensors)  ->  CUDA tensor buffers
      ->  transformer primitives (norm, rope, gemm, attention, ...)  ->  forward
```

The reference implementation of every primitive is validated against a
**golden fixture** captured once from the official HiDream/Transformers oracle.
This enforces an engineering discipline where the Python upstream is treated
strictly as an oracle — a source of truth for semantics — rather than as a
runtime dependency.

## Core principles

- **Python is an oracle, not a runtime.** The official upstream checkout lives
  at `/python` (relative to the project root), is read-only and git-ignored,
  and is never patched. Production inference is pure C/CUDA.
- **Reproducibility is contractual.** Revisions, versions, hashes, environment
  facts, and gate evidence are persisted to files (locks, manifests, run
  ledger); a branch name is never a substitute for an immutable commit.
- **Zero-forward validation during development.** M0–M1.2 forbid transformer
  forward passes and image generation. Primitives are validated at the
  metadata/dtype/GEMM level against frozen golden tensors (validation levels
  V0–V2). Full model runs are scarce validation resources, not a debugging
  loop.
- **Dev/Base symmetry.** The Dev and Base/Full model profiles stay supported
  through one shared configuration and code path.
- **Ownership boundaries.** Only engine code (`src/`, `tools/`, `config/`,
  `docs/`, `scripts/`) is versioned. Oracle checkout, model weights, HF caches,
  build output, and generated runtime data are ignored.

A fuller statement of these invariants lives in
[`docs/WORKING_MODE.md`](docs/WORKING_MODE.md).

## Repository layout

```
include/      Public C ABI header(s)          (hidream.h)
src/          C/CUDA engine sources
  io/         JSON, safetensors, sha256 readers
  model/      Model profile, config, weight ingestion
  cuda/       Reference CUDA primitives (norm, rope, gemm, attn, act)
  main.c      Engine entry point
tests/unit/   C test harnesses (model loader, weights, primitives)
tools/        Python oracle helpers (freeze, capture, guard/ledger)
config/       Versioned locks and model profiles (oracle.lock, models.lock,
              dev.json, base.json, startup/tensor manifests)
scripts/      Bootstrap scripts (checkout oracle, download models, freeze)
docs/         Milestone specs, working mode, status, golden contracts
artifacts/    Git-ignored runtime evidence (env, oracle, models, golden)
build/        Git-ignored build output
/python/      Git-ignored oracle checkout (upstream, read-only)
/models/      Git-ignored model weights (dev/, base/)
```

`artifacts/` holds all reproducible evidence and is git-ignored by design;
only compact, intentionally-versioned manifests are committed into `config/`.

## Configuration & locking

- **Oracle pin** — [`config/oracle.lock`](config/oracle.lock) freezes the
  upstream HiDream repository URL, branch, and exact commit SHA.
- **Model pins** — [`config/models.lock`](config/models.lock) pins each model
  profile to an immutable Hugging Face revision and records the local path and
  download status. Dev is downloaded; Base/Full may be configured but not yet
  downloaded.
- **Model profiles** — `config/dev.json` / `config/base.json` carry dtype,
  step count, and topology. Dev and Base are symmetric profiles over one
  shared code path.
- **Startup freeze** — `config/startup_manifest_dev.json` captures a canonical
  prompt, token IDs, timesteps, scheduler/config metadata.
- **Tensor manifest** — `config/tensor_manifest_dev.json` records the structure
  of every state-dict tensor (name, shape, dtype, numel) without parameter
  values.

## Build & test

A CUDA 13 toolchain and cuBLAS (`libcublas.so.13` / `libcublasLt.so.13`) are
required; the target is NVIDIA GB10 compute capability 12.1 (`-arch=sm_121`).

```sh
make            # build the engine (build/hidream)
make test       # run the C test suites (model loader + weights)
make test-primitives   # M1.2 reference CUDA primitive tests (when wired)
make clean
```

The build is gcc-based; the `src/cuda/` primitives compile with `nvcc` and are
linked into the test/engine binaries via the `Makefile` CUDA rules.

## Validation workflow

The golden-fixture pipeline (V2 level) is:

1. **Oracle checkout** — `scripts/m0/01_checkout_oracle.sh` pins the official
   upstream revision under `/python` (read-only).
2. **Model acquisition** — `scripts/m0/02_download_models.sh dev` downloads Dev
   weights into git-ignored `models/dev` and writes `config/models.lock`.
3. **Startup freeze** — `scripts/m0/03_freeze_startup.sh dev` captures the
   canonical freeze manifest without any forward pass.
4. **Fixture capture** — `tools/capture_m1_2_fixtures.py` produces deterministic
   golden tensors for each primitive (single model load, no forward, no
   generation) under `artifacts/m1/golden/`.
5. **Guard + ledger** — `tools/m1_guard.py` verifies offline environment,
   exact versions, and oracle cleanliness, and records every run in
   `artifacts/m1/run_ledger.jsonl` with validation level and forward count.

Every gate leaves durable evidence and is recorded in
`docs/M1_STATUS.md`; the reproducibility contract is defined in
`docs/M1_2_GOLDEN_CONTRACT.md` and `docs/M1_NUMERICAL_CONTRACT.md`.

## Contributing / engineering etiquette

- Prefer existing scripts over ad-hoc shell sequences; improve an engine-side
  bootstrap script if reproducibility requires it, never the oracle.
- Never commit model weights, caches, or large generated artifacts.
- Never replace an immutable revision with a branch name.
- Keep each commit a single logical gate, and run the tests before check-in;
  keep `docs/*_STATUS.md` current.
- If unsure whether an action counts as model execution, treat it as forbidden
  until proven startup-only.

## Milestones

See [`MILESTONES.md`](MILESTONES.md) for the full specification. In short:
M0 pins the oracle/model/startup; M1 loads weights and implements the reference
CUDA transformer primitives with V2 golden validation; later milestones build
the full deterministic forward and image generation.