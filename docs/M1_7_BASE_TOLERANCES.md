# M1.7 — Base Compatibility One-Forward: Quantitative Tolerances

**Status:** FROZEN (defined before the Base V3 gate runs, per MILESTONES.md §M1.7)

**Milestone:** M1.7 — M1 closure: Base compatibility
**Validation level:** V3 (one oracle run + one native run, non-iterative)
**Commit subject (gate):** `test(m1): validate base one-forward compatibility against frozen oracle`

---

## 1. Scope

This document defines, **before** the gate executes, the quantitative tensor
tolerances for the single controlled Base one-forward at the validation
resolution (64×64, 4 image tokens, seq 23) with the canonical prompt/seed:

- prompt: `a red fox sits under a cherry blossom tree`
- seed: `42`
- model profile: `base` (HiDream-ai/HiDream-O1-Image, revision
  `0b0901d99f200389e138c61946af1185f5f49a13`)
- timestep domain: **model_timestep = 0.001** (scheduler_timestep 999 →
  sigma 0.999 → model_timestep 1−sigma ≈ 0.001 → embedder input ≈ 1.0),
  per docs/M1_FORWARD_CONTRACT.md §15.1. This is the CORRECT semantics; the
  M1.4 fixture (M1_V3_DEV_FORWARD_0) is LEGACY/INVALID for pipeline
  semantics and is NOT used as the Base reference.
- no scheduler loop, no CFG, no image generation: exactly one transformer
  forward at model_timestep = 0.001.

The native run replays the frozen golden inputs verbatim and compares every
captured checkpoint in strict order (model input → embedding → target
embedding → timestep conditioning → block 0 → mid block → last block →
final norm input → final norm output → final head → complete output).

## 2. Reference measurements (input to this spec)

All from the frozen oracle / prior milestones (no new runs):

| Source | Quantity | Value |
|--------|----------|-------|
| M1.4 golden (V3, 1 forward, Dev) | block_last NRMSE / cos | 0.0246 / 0.99996 |
| M1.4 golden (V3, 1 forward, Dev) | complete_output envelope | NRMSE ≤ 0.18, cos ≥ 0.99 |
| contract §4 class D | attention / block | NRMSE ≤ 1e-2, cos ≥ 0.999 |
| contract §4 class C | GEMM/projection | NRMSE ≤ 5e-3, cos ≥ 0.9999 |

Base topology is identical to Dev (hidden 4096, 36 layers, ff 12288, heads
32, kv 8, head_dim 128) — only the weights differ. The same native machinery
(`hd_forward` + weight store) is exercised; the gate proves the shared
implementation path handles Base weights with the same numerical behavior.

## 3. Tensor tolerances (native vs golden, strict order)

The comparison stops at the first meaningful divergence (strict order per the
M1.4 contract). Bounds are the M1.4 whole-forward gate bounds
(docs/M1_NUMERICAL_CONTRACT.md §6), re-derived from the same amplification
curve; they are NOT tuned to the observed Base run.

| Checkpoint | Bound | Basis |
|------------|-------|-------|
| embedding / target_embedding / timestep_conditioning | NRMSE ≤ 5e-3, cos ≥ 0.9999 | class C (GEMM/projection) |
| block_0 / block_mid | NRMSE ≤ 1e-2, cos ≥ 0.999 | class D (attention/block) |
| block_last / final_norm_input | NRMSE ≤ 3e-2, cos ≥ 0.999 | accumulated 36-layer bf16 drift; tail algebra proven correct |
| final_norm / final_head_input | NRMSE ≤ 0.1, cos ≥ 0.99 | 3.54× amplification of block_last ≤ 3e-2 |
| complete_output | NRMSE ≤ 0.18, cos ≥ 0.99 | 5.73× amplification of block_last ≤ 3e-2 |

Additional structural assertions (exact, class A):

- `01_model_input` int64[1,19] — native input_ids must equal the golden
  bytes (same canonical prompt tokenization).
- `scheduler_timestep` = 999.0, `sigma` = 0.999, `model_timestep` ≈ 0.001,
  `timestep_embedder_input` ≈ 1.0 — the native run must consume
  model_timestep (NOT scheduler time); the manifest metadata must agree
  with the golden (guards against timestep-domain regression).
- no NaN/Inf in any compared tensor.

## 4. Failure conditions

The gate FAILS if:

- the golden capture's timestep-domain sanity check fails
  (model_timestep ≠ 1 − 999/1000, or embedder input ≠ ≈1.0);
- the native run cannot load Base weights (config + weight loading
  compatibility broken);
- any strict-order checkpoint exceeds its bound;
- the native run consumes scheduler time (999) instead of model time
  (0.001) — timestep-domain regression;
- the single oracle run fails or produces non-finite tensors.

## 5. Run budget

- Oracle: exactly 1 V3 whole-model forward (the single allowed Base oracle
  run). No re-run.
- Native: 1 replay run.
- No V6 (50-step Base generation) is required to close early M1
  (MILESTONES.md §M1.7).