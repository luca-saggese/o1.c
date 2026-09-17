# M2 cuBLAS Freeze

## Performance

| Path | Dev 1024² / 28 steps |
|------|----------------------|
| Python legacy | ~90 s |
| Native reference GEMM | ~25 min estimated / impractical |
| Native cuBLAS | ~27 s |

| Metric | Value |
|--------|-------|
| Approx speedup vs Python | ~3.3× |
| Approx speedup vs old native | ~50×+ |

Measured on NVIDIA GB10 (DGX Spark), BF16, scheduler flash, seed 42,
guidance 0.0, shift 1.0. Native cuBLAS run: DENOISE ~24.6 s,
BLOCK_SINGLE ~22 ms/block, MODEL_LOAD ~36 s.

## Known numerical discrepancy

| Backend | `test_full_forward` |
|---------|---------------------|
| Reference | 14/14 PASS |
| cuBLAS | 13/14 PASS |

Remaining failure: `complete_output`

| Metric | Reference | cuBLAS | Threshold |
|--------|-----------|--------|-----------|
| cosine | ~0.99085 | ~0.9888 | 0.99 |

Additional observations:

- isolated final-head reference vs cuBLAS are numerically equivalent
- discrepancy emerges progressively across transformer blocks
- final generated image is visually close but slightly different
- no gross semantic/layout corruption observed
- tolerance was NOT relaxed
- golden fixtures were NOT regenerated

## Disposition

```
STATUS: ACCEPTED_FOR_M2_PERFORMANCE FREEZE
PARITY ISSUE: OPEN
BLOCKER FOR CURRENT PERFORMANCE WORK: NO
FOLLOW-UP: REQUIRED BEFORE FINAL RELEASE / NUMERICAL CLOSURE
```

## Follow-up investigation

- direct reference-vs-cuBLAS same-input comparison per layer
- identify projection family responsible for accumulated drift
- evaluate cuBLASLt algorithm selection
- evaluate selective stricter GEMM policy if needed
- compare final image suite across fixed seeds

Not executed during this freeze; deferred to numerical closure.