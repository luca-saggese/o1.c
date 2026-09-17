# HiDream O1 Image Inference Engine (`o1.c`)

A from-scratch **native C/CUDA inference engine** for the
[HiDream O1 Image](https://huggingface.co/HiDream-ai) foundation image model,
targeted at the **NVIDIA GB10** (DGX Spark) platform: compute capability 12.1
(`-arch=sm_121`), CUDA 13.0 toolchain. The engine re-implements the model's
transformer math in hand-written CUDA kernels, validated tensor-by-tensor
against a frozen Python oracle, so that **production inference never depends
on Python**.

> **Status:** Milestones **M0, M1, M1.1, M1.2, and M1-post are COMPLETE** —
> the production native T2I path is working end-to-end (28-step Dev
> generation at 1024², PNG output, no Python at runtime). **M2 (performance /
> pre-baseline) is in progress**: cuDNN SDPA attention and a persistent cuBLAS
> GEMM production backend are frozen, with the numerical discrepancy
> documented in [`docs/M2_CUBLAS_FREEZE.md`](docs/M2_CUBLAS_FREEZE.md). See
> [`MILESTONES.md`](MILESTONES.md) for the milestone ladder,
> [`docs/M1_POST_CLOSEOUT.md`](docs/M1_POST_CLOSEOUT.md) for the feature
> closeout table, and [`docs/M1_POST_STATUS.md`](docs/M1_POST_STATUS.md) for
> execution state.

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

Measured on GB10 (Dev 1024², 28 steps, BF16):

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
  --ref-image PATH        reference image (repeatable, edit/personalize)
  --width N               output width (default: 1024)
  --height N              output height (default: 1024)
  --steps N               inference steps (default per profile)
  --seed N                RNG seed (default: 123456)
  --scheduler flash|default|flow_match   (default per profile)
  --guidance F            CFG scale (default per profile)
  --shift F               scheduler shift (default per profile)
  --noise-start F         noise_scale_start (default 8.0)
  --noise-end F           noise_scale_end (default 8.0)
  --noise-clip F          noise_clip_std (default 8.0)
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

## Test suite

`make test-*` targets (all C/CUDA, no Python):

| Target | Covers |
|--------|--------|
| `test-rng` | native RNG parity vs oracle |
| `test-png` / `test-image` | PNG encode/decode, image I/O |
| `test-layout` | layout conditioning sequence (bit-exact) |
| `test-seq` / `test-seq-ref` | sequence builder parity vs M1.4 fixture / oracle ground truth |
| `test-seq-diag` | sequence manifest diagnostics + workspace estimate |
| `test-seq-profiles` | frozen per-mode sequence geometry (perf freeze §70) |
| `test-decode` | output decode (unpatchify) |
| `test-refiner` | prompt refiner clients |
| `test-progress` / `test-preview` | progress callback / preview extraction |
| `test-sanity` | native 1024² runner (1-step decode + PNG) |
| `test-m17` | M1.7 full-forward numerical gate (incl. `test-m17-base`) |
| `test-tokenizer` | native tokenization (incl. long/multilingual text) |
| `test-gemm-smoke` | deterministic cuBLAS GEMM mapping check |
| `test-layer-replay` | layer-by-layer reference-vs-cuBLAS chain comparison |
| `test-final-head` | final-head GEMM decisive test (5 configs) |
| `test-gguf` | GGUF pack header parse + payload round-trip vs safetensors |
| `bench-block` | per-stage decoder block benchmark (CUDA events) |

## Milestone status

| Milestone | Status | Summary |
|-----------|--------|---------|
| M0 | ✅ COMPLETE | Oracle/model/startup pinned; reproducible environment |
| M1 | ✅ COMPLETE | Native profiles, weight ingestion, CUDA primitives, forward, scheduler, tokenizer, full 28-step Dev inference validated |
| M1.1 | ✅ COMPLETE | Weight ingestion into deterministic CUDA buffers |
| M1.2 | ✅ COMPLETE | Reference CUDA transformer primitives (V2 golden parity) |
| M1-post | ✅ COMPLETE | Production T2I path, feature parity (editing, multi-ref, layout, long text), perf freeze; closeout finishing |
| M2 | 🚧 IN PROGRESS | Performance / pre-baseline: timing instrumentation, benchmark harness, cuDNN SDPA attention, persistent cuBLAS GEMM backend (frozen) |
| M3 | ⏳ planned | Production loader / memory optimization |

Known limitation: **2048×2048 is `BLOCKED_M2`** — semantics are correct and
shape-generic, but the materialized O(S²) attention workspace (≈2.17 GB for
scores+probs at S=4115, ≈2.85 GB total) exceeds the GB10 budget. Fixing it
(flash/chunked attention) is an M2 item. See
[`docs/M1_POST_PERF_FREEZE.md`](docs/M1_POST_PERF_FREEZE.md) for the frozen
sequence geometry table and
[`docs/M1_POST_CAPABILITY_MATRIX.md`](docs/M1_POST_CAPABILITY_MATRIX.md) for
the full capability × profile matrix.

## M2 direction

M2 is the performance/pre-baseline milestone: measure before optimizing.
Current state (see [`docs/M2_CUBLAS_FREEZE.md`](docs/M2_CUBLAS_FREEZE.md)):

- **cuDNN SDPA attention backend** — replaces the eager reference attention in
  the production path (BF16/GQA, FlashAttention-2 semantics); reference
  attention kept for correctness comparison.
- **Persistent cuBLAS GEMM backend** — `cublasGemmEx` (BF16 in/out, FP32
  accumulate) with handles created once at runtime init; the hand-written
  reference GEMM (0.28 TFLOP/s) is kept for correctness/debug only.
- **Pipelined weight loader** — one aligned CUDA arena, pinned staging,
  dedicated nonblocking upload stream; safetensors load drops from 30 s to
  27 s, and the materialized GGUF pack (see above) to ~5 s.
- **Frozen performance** — native Dev 1024²/28-step generation ≈27 s vs
  ≈90 s Python legacy (≈3.3× speedup) and ≈25 min estimated for the old
  reference GEMM (≈50×+). A known numerical discrepancy (13/14 full-forward
  PASS on the cuBLAS path) is documented and accepted for the performance
  freeze; numerical closure is a required follow-up before final release.

Planned work (see `docs/M2_CANDIDATES.md`):

- **Bottleneck profiling** (Nsight Systems/Compute): rank GEMM, attention,
  normalization, elementwise kernels, launches, memory movement.
- **Low-risk execution optimizations**: cuBLASLt plan caching, CUDA Graph
  capture, kernel fusion, flash/chunked attention (unblocks 2048²), shape
  specialization.
- **Precision**: BF16/FP32-native path first; FP8/FP4 only later, each with
  its own numerical/quality contract.
- **Quality suite**: fixed prompt corpus with perceptual + image metrics,
  prerequisite for any quality-affecting precision work.

Every optimization must keep the M1 correctness gates green (V2/V3 before any
V5 rerun).

## References

- [`MILESTONES.md`](MILESTONES.md) — milestone ladder and validation cost ladder
- [`docs/M1_POST_STATUS.md`](docs/M1_POST_STATUS.md) — M1-post execution state
- [`docs/M1_POST_CLOSEOUT.md`](docs/M1_POST_CLOSEOUT.md) — feature closeout table
- [`docs/M1_POST_CAPABILITY_MATRIX.md`](docs/M1_POST_CAPABILITY_MATRIX.md) — capability × profile matrix
- [`docs/M1_POST_PERF_FREEZE.md`](docs/M1_POST_PERF_FREEZE.md) — frozen sequence geometry (§70)
- [`docs/M1_POST_SCHEDULER_MATRIX.md`](docs/M1_POST_SCHEDULER_MATRIX.md) — scheduler recipes
- [`docs/M2_CANDIDATES.md`](docs/M2_CANDIDATES.md) — M2 optimization backlog
- [`docs/M2_CUBLAS_FREEZE.md`](docs/M2_CUBLAS_FREEZE.md) — cuBLAS performance freeze + numerical discrepancy
- [`docs/WORKING_MODE.md`](docs/WORKING_MODE.md) — engineering invariants
- Upstream: [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai)

No license file is shipped in this repository; model weights and upstream
code remain subject to their own licenses.
