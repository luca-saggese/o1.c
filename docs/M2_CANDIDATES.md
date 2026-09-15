# M2 Candidates

Deferred optimization / architecture backlog for M2, kept out of M1.3a scope
per `docs/M1_3A_PRE_FORWARD_ARCHITECTURE_FREEZE.md` §64. M1.3a only freezes
the constraints that keep these feasible; none are implemented now.

## Performance / execution

- **CUDA Graph** capture of the device-resident forward (M1.3a keeps the
  architecture capture-friendly: stable buffers, profile-fixed shapes,
  persistent streams/handles).
- **cuBLASLt tuning** — persistent handle tuning, layout/transpose tuning.
- **FlashAttention / custom attention** — replace the reference materialized
  `O(S²)` attention (scores+probs) needed for DEV-2048 production
  (see `docs/M1_MEMORY_LIFETIMES.md` §12; scores+probs ≈ 2.17 GB at 2048).
- **Kernel fusion** — e.g. residual+norm, q/k/v head-split+RoPE, MLP gated
  product.
- **Shape specialization** — profile-fixed variants of hot kernels.
- **Bandwidth optimization** — attribute/GB-s analysis; the M1.3a tensor
  descriptors already carry bytes-read/written and FLOP accounting hooks.
- **Scheduler device migration** — move scheduler_step onto device if measured
  to be host-bound.
- **Microbenchmarking** of individual primitives.

## Precision

- **FP8 / FP4 / NVFP4** and weight/activation quantization — must reuse the
  same semantic execution path and require their own numerical/quality
  contract. Quality suite (10–20 prompts) is a prerequisite before any
  aggressive quantization.

## Quality

- **Quality suite** — a fixed prompt corpus with perceptual + image metrics,
  frozen reference resolution, needed before quantization or
  quality-affecting precision work.

## Memory

- Reduce per-forward activation peak; evaluate chunked/flash attention to drop
  the O(S²) attention workspace (blocking 2048 production).