# Performance

This page summarises measured performance for the native o1.c engine. The
full measurement ledgers, methodology and per-kernel breakdowns live in
[`docs/M2_PERFORMANCE.md`](M2_PERFORMANCE.md) and the freeze documents.

## Test platform

| Item | Value |
|------|-------|
| GPU | NVIDIA GB10 (DGX Spark), compute capability 12.1 |
| Driver | 580.126.09 |
| CUDA | 13.0 (V13.0.88) |
| cuBLAS | 13.1.0.3 |
| cuDNN | 9.20 |
| Compiler | gcc 13.3.0, nvcc `-arch=sm_121 -O2 -std=c++17` |
| Precision | BF16 weights and activations, FP32 accumulate |

## Canonical end-to-end numbers

Measured with the production CLI, timing instrumentation disabled, model load
excluded from the inference number.

| Workload | Recipe | Wall clock |
|----------|--------|-----------|
| Dev-2604 text-to-image | 2048×2048, 28 steps, seed 42 | ≈ 86 s |
| Dev-2604 edit | 2048×2048, 28 steps, 1 reference, seed 42 | ≈ 196 s |
| Base text-to-image | 2048×2048, 50 steps, seed 42, CFG | ≈ 320 s |

These are the reference points; any optimization is judged against them.

## Where the time goes

**Text-to-image (2048×2048).** Sequence length `S = 4115` (19 text tokens +
4096 image tokens) over 36 transformer blocks. GEMMs dominate and already run
at roughly 94 % of the measured device GEMM ceiling, so the remaining target
is attention.

**Edit (2048×2048 with one reference).** The reference's pixel tokens are
appended to the denoiser sequence, so `S = 8278` (150 text + 4096 target +
4032 reference). Per-step cost roughly doubles versus text-to-image. The
breakdown of the denoise loop is approximately:

| Component | Share |
|-----------|-------|
| GEMMs (q/k/v/o, gate/up/down, head) | ≈ 61 % |
| SDPA | ≈ 27 % (after the two-pass change) |
| Other block kernels, embedding, scheduler | remainder |
| Vision tower (one-shot, reference encoding) | ≈ 0.4 % |

## Optimizations applied

| Change | Effect |
|--------|--------|
| cuBLASLt-backed persistent GEMM | GEMM at ≈ 94 % of device ceiling |
| Two-pass SDPA (text-to-image) | SDPA 14.6 → 11.7 ms/block, ≈ 20 % |
| Two-pass SDPA (edit decoder) | SDPA 53.6 → 42.9 ms; denoise 181.2 → 170.0 s |
| Vision hot-path cleanup | removed per-call allocations and syncs |
| Reference-embedding cache | avoids recomputing `x_embedder` for static reference rows |

## What is already at the limit

- **GEMM.** The production GEMM backend reaches ≈ 94 % of the measured device
  GEMM ceiling. Further GEMM work is closed.
- **Full attention pass.** After the two-pass change, the image-region pass is
  irreducible full non-causal attention over the whole sequence.

## Notes on measurement

- GPU timings use CUDA events recorded around work and resolved after
  execution — never CPU wall clock around asynchronous kernels.
- Model loading is measured and reported separately from inference so it does
  not contaminate the generation number.
- Benchmarks are reproducible: fixed prompt, seed 42, identical binaries for
  A/B comparisons, warm-up runs excluded.

## Further reading

- [`docs/M2_PERFORMANCE.md`](M2_PERFORMANCE.md) — full measurement ledger
- [`docs/M2_CUBLAS_FREEZE.md`](M2_CUBLAS_FREEZE.md) — GEMM freeze
- [`docs/M1_POST_PERF_FREEZE.md`](M1_POST_PERF_FREEZE.md) — frozen sequence geometry
- [`docs/RELEASE_VALIDATION.md`](RELEASE_VALIDATION.md) — release validation runs
