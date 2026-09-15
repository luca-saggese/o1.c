# M1 Numerical Comparison Contract

This document is the binding numerical contract for all M1 native-vs-oracle
comparisons. It is finalized **before** evaluating native numerical results.
Tolerances are guardrails, not permission for unexplained systematic error.

## 1. Reference inputs (fixed for the whole milestone)

- **Canonical prompt:** `a red fox sits under a cherry blossom tree`
- **Canonical text token IDs (19):**
  `[151644, 872, 198, 64, 2518, 38835, 23011, 1212, 264, 40880, 88758, 4916, 151645, 198, 151644, 77091, 198, 151669, 151673]`
- **Seed:** `42` (noise derived from `seed+1=43`; RNG reseeded with `seed+1`)
- **Scheduler:** Dev T2I non-editing → `flash` (stochastic flow-match SDE)
  - `num_inference_steps = 28`, `shift = 1.0`
  - `DEFAULT_TIMESTEPS = [999,987,974,960,945,929,913,895,877,857,836,814,790,764,737,707,675,640,602,560,515,464,409,347,278,199,110,8]`
  - `noise_scale_start = 8.0`, `noise_scale_end = 8.0`, `noise_clip_std = 8.0`
  - `T_EPS = 0.001`, `PATCH_SIZE = 32`
- **FAST_VALIDATION_RES:** `64×64` (see section 4)
- **Compute dtype:** BF16 forward under `torch.autocast`; weights loaded FP32.

## 2. Comparison procedure

For floating tensors, cast both sides to FP32 for analysis and record **all**
of:

1. shape equality
2. dtype / source dtype
3. max absolute error
4. mean absolute error
5. RMSE
6. normalized RMSE (NRMSE = RMSE / RMS(reference))
7. cosine similarity
8. RMS(reference)
9. RMS(candidate)
10. NaN count
11. Inf count

Max relative error is **not** a primary metric (values near zero are
misleading).

Token IDs, integer indices, tensor names, shapes, scheduler step counts and
masks must be **exact** (Class A).

## 3. Tolerance classes

These are starting contracts; tighten where possible.

| Class | Scope | Bound |
|-------|-------|-------|
| **A** | deterministic/indexing/layout: token IDs, shape transforms, patchify/unpatchify, head split/merge, mask construction, position-id construction | **exact** |
| **B** | simple BF16/FP32 pointwise: RMSNorm, Q/K norm, RoPE/MRoPE, SiLU, elementwise scheduler arithmetic | NRMSE ≤ 2e-3, cosine ≥ 0.99999, no NaN/Inf mismatch |
| **C** | GEMM/projection outputs | NRMSE ≤ 5e-3, cosine ≥ 0.9999 |
| **D** | attention / block outputs | NRMSE ≤ 1e-2, cosine ≥ 0.999 |
| **E** | complete transformer forward | NRMSE ≤ 2e-2, cosine ≥ 0.995 |

Rules:

- A large localized max error, NaN, mask mismatch, transposition error, wrong
  RoPE, wrong GQA mapping, or wrong output scale is a **correctness bug** even
  if a coarse aggregate metric passes.
- Tolerances may only be changed with a written technical reason identifying
  the source of legal numerical variation, and must be a separate commit or
  explicitly documented in the gate commit.

## 4. FAST_VALIDATION_RES justification (static, no generation)

- `PATCH_SIZE = 32`; `image_len = (h//32)*(w//32)`; resolution must be a
  multiple of 32.
- `32×32` → 1 patch → 1 image token → crashes
  `python/models/utils.py::get_rope_index_fix_point`
  (`vision_start_indices + 1` indexing; `IndexError: index 20 out of bounds
  for size 20`).
- `64×64` → 2×2 patches → 4 image tokens → total sequence 23
  (19 text + 4 image). **This is the minimum structurally valid resolution.**
- 2048×2048 is not the routine M1 correctness resolution.

## 5. Full-output (V5) comparison

Final full-output bound is set from observed V3/V4 drift **before** viewing
the V5 native result:

- raw tensor NRMSE / cosine / max|err| / mean|err|
- PSNR, SSIM (if available), pixel/channel range
- NaN/Inf counts, image dimensions

Raw final float tensor is compared first; image encoding second. PNG byte
identity is not required. "Looks similar" is supplementary only.
