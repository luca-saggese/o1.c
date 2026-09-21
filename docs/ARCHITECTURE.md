# Architecture

o1.c is a from-scratch native C/CUDA inference engine for the HiDream-O1-Image
foundation model. It re-implements the transformer math in hand-written CUDA
kernels so that **production inference never depends on Python**.

## Execution path

A normal CLI generation traverses:

```
build/hidream                       (CLI, src/main.c)
  -> hd_generate                    (src/runtime/generate.c)
       sequence construction        (src/runtime/sequence.c)
       scheduler                    (src/model/scheduler.c)
       denoise loop
         -> hd_forward              (src/model/forward.c)
              transformer blocks    (src/model/block.c)
                RMSNorm             (src/cuda/norm.cu)
                q/k/v/o projections (src/cuda/gemm.cu)
                RoPE                (src/cuda/rope.cu)
                SDPA                (src/cuda/hd_cudnn_sdpa.cu)
                gate/up/down + SwiGLU (src/cuda/gemm.cu, src/cuda/act.cu)
                residual/pointwise  (src/cuda/residual.cu)
              final norm, final head
         -> decode / unpatchify      (src/runtime/decode.c, src/image/)
         -> PNG write                (src/image/, src/io/)
```

Edit and personalization modes insert reference images into the same path:
the reference is patch-embedded once and its pixel tokens are appended to the
denoiser sequence, so the transformer sees a longer `S` (see the performance
notes for the measured geometry).

## Source layout

```
include/      Public C ABI header (hidream.h)
src/          C/CUDA engine sources
  main.c      CLI entry point
  model/      Profiles, config, weights, tokenizer, scheduler, forward, block
  runtime/    Request, sequence builder, generate, decode, engine, timing
  cuda/       Hand-written CUDA kernels (norm, rope, gemm, attn, act, embed,
              residual, sched, support, vision, lora_merge)
  io/         JSON, safetensors, GGUF, sha256 readers, PNG wrapper
  image/      Image layout and PNG encode/decode
  server/     OpenAI-compatible HTTP server (not the performance target)
tests/        C test harnesses (unit + integration) and server smoke scripts
tools/        Python oracle helpers (freeze, capture, GGUF convert)
config/       Versioned locks and profiles (oracle.lock, models.lock, dev.json,
              base.json)
scripts/      Setup, model download, release model build
third_party/  Vendored cudnn-frontend headers
docs/         Engineering documentation
```

## Key design decisions

- **Python is an oracle, not a runtime.** The frozen upstream checkout is used
  only for offline validation; the shipped engine is pure C/CUDA.
- **Reproducibility is contractual.** Revisions are pinned to immutable commit
  SHAs in `config/`; a branch name is never a substitute.
- **One code path for Dev and Base.** They share the implementation and differ
  only in profile parameters (steps, guidance, shift, scheduler).
- **Deterministic weight ingestion.** Weights load into one aligned CUDA arena
  through pinned staging on a dedicated non-blocking stream.
- **cuDNN SDPA for attention.** The production attention uses cuDNN's fused
  scaled-dot-product-attention with persistent plans and workspace; the eager
  reference attention is retained for correctness comparison.
- **Persistent cuBLAS GEMM.** `cublasGemmEx` with handles created once; the
  hand-written reference GEMM is kept for correctness/debug only.

## Attention masking model

The sequence is `[text tokens][image tokens]`, and for reference/edit modes
`[text tokens][target image tokens][reference pixel tokens]`. Attention is
causal over the text (autoregressive) region and full over the image region.
This is exactly the shape that the two-pass SDPA formulation exploits: one
causal pass over the AR rows and one full-attention pass over the image rows.

## Further reading

- [`docs/DEVELOPMENT.md`](DEVELOPMENT.md) — milestones and engineering history
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — measured performance
- `docs/M1_EXECUTION_ARCHITECTURE.md` — original execution architecture
- `docs/M1_POST_PERF_FREEZE.md` — frozen sequence geometry
