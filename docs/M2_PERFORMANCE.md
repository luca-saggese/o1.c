# M2 Performance Ledger — Native CLI Inference Path

Status: **measurement / instrumentation only. No optimization implemented.**
Correctness and model behavior are frozen. This document is the comparable
baseline for all subsequent M2 work.

---

## 1. Repository state

| Field | Value |
|---|---|
| Branch | `main` |
| HEAD | `0ad7f18833eecdd4b2035dc229e954bb1110c084` |
| Working tree at start | clean (no uncommitted modifications) |
| Working tree now | instrumentation only (see §16) |

Instrumentation changes (uncommitted, no optimization):

```
 M Makefile
 M src/main.c
 M src/model/block.c
 M src/model/forward.c
 M src/runtime/o1_timing.c
 M src/runtime/o1_timing.h
?? tests/unit/bench_gemm.c
```

---

## 2. Canonical CLI path

A normal CLI T2I generation traverses exactly:

| Stage | File / function |
|---|---|
| CLI executable | `build/hidream` |
| CLI entry point | `src/main.c` `main()` (line 206; gen path ~340) |
| Generation entry point | `src/runtime/generate.c` `hd_generate()` (line 1442) |
| Engine | `src/runtime/engine.c` |
| T2I generation | `src/runtime/generate.c` `hd_engine_generate_t2i()` (~930) |
| Denoise loop | `src/runtime/generate.c` lines 755–836 |
| Transformer forward | `src/model/forward.c` `hd_forward()` (line 190) |
| Block | `src/model/block.c` `hd_decoder_block()` |
| GEMM dispatch | `src/cuda/gemm.cu` `hd_linear()` → `hd_gemm_cublas()` |
| SDPA dispatch | `src/model/block.c` → `src/cuda/hd_cudnn_sdpa.cu` `hd_sdpa_execute()` |
| Scheduler | `src/model/scheduler.c` + `src/cuda/sched.cu` |
| Reconstruction | `src/runtime/generate.c` lines 840–889 |

The HTTP server is **not** on this path and is out of scope for M2.

---

## 3. Environment

| Field | Value |
|---|---|
| Hardware | NVIDIA GB10 (DGX Spark), unified memory, 121 GiB system RAM |
| Driver | 580.126.09 |
| CUDA | 13.0 (V13.0.88) |
| cuBLAS | 13.1.0.3 (`/usr/local/cuda/lib64/libcublas.so.13.1.0.3`) |
| cuDNN | 9.20.0.48 (pip `nvidia-cudnn-cu13`, `/home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn/lib`) |
| Compiler | gcc 13.3.0 (Ubuntu 24.04) |
| nvcc | `-arch=sm_121 -O2 -std=c++17` |
| C flags | `-O2 -g -std=c11 -Wall -Wextra` |
| cuDNN wrapper | built `-O0` (vendored cudnn-frontend v1.22.1) |
| Profilers | `nsys` / `ncu` / `compute-sanitizer` at `/usr/local/cuda/bin` |

`LD_LIBRARY_PATH` must include the cuDNN pip lib dir at runtime.

---

## 4. Canonical benchmark configuration (frozen)

| Field | Value |
|---|---|
| Model | Dev (`artifacts/models/hidream-o1-dev-bf16.gguf`, 17.61 GB) |
| Resolution | 2048×2048 |
| Steps | 28 |
| Seed | 42 |
| Scheduler | `flash` (production default) |
| GEMM backend | cuBLAS (`cublasGemmEx`, `CUBLAS_GEMM_DEFAULT`) |
| Precision | bf16 |
| Guidance | 0.0, shift 1.0 |
| References | none |
| Prompt (verbatim) | `A dog holds a sign that says HiDream-O1-Image release.` |

### Production build command

```bash
make all          # -> build/hidream  (no timing code compiled in)
```

### Canonical CLI command

```bash
LD_LIBRARY_PATH=/home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH \
./build/hidream --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --prompt "A dog holds a sign that says HiDream-O1-Image release." \
  --width 2048 --height 2048 --steps 28 --seed 42 \
  --no-progress --output artifacts/m2/perf/baseline/run1.png
```

### Timing methodology

* CPU phases: `O1_TIMING_BEGIN/END` (compile-time gated by `-DO1_DEBUG_TIMING`).
* GPU phases: `O1_TIMING_BEGIN_GPU/END_GPU` — CUDA events recorded around work,
  **resolved only in the report** (no per-op synchronization).
* No CPU wall clock is used as a proxy for GPU time.
* Production binary contains zero timing code; all timing uses
  `build/hidream_timing` / `build/hidream_timing_block`.

---

## 5. Canonical baseline — 3 full CLI runs

1 warm-up + 3 measured identical generations (28 steps, 2048², seed 42).

| Run | Wall clock | Max RSS | Result |
|---|---|---|---|
| warm-up | 1:25.55 (85.55 s) | 823 MB | PASS |
| run 1 | 1:26.00 (86.00 s) | 823 MB | PASS |
| run 2 | 1:24.11 (84.11 s) | 823 MB | PASS |
| run 3 | 1:24.08 (84.08 s) | 823 MB | PASS |

**Baseline end-to-end: 84.1–86.0 s** (median ≈ 85.0 s).

### Load time separated from inference time

From the instrumented 1-step and 3-step runs (same config, 1/3 steps):

| Phase | 1-step | 3-step | Notes |
|---|---|---|---|
| MODEL_LOAD | 5.679 s | 6.294 s | **excluded from inference** |
| PROMPT_TOKENIZE | 0.038 s | 0.048 s | |
| INITIAL_NOISE | 0.253 s | 0.257 s | |
| INPUT_PREPARE | 0.685 s | 1.021 s | includes mask build + H2D |
| DENOISE_TOTAL | 3.460 s | 8.767 s | 2.92 s/step |
| OUTPUT_RECONSTRUCTION | 0.117 s | 0.144 s | |
| REQUEST_TOTAL | 10.391 s | 16.730 s | |
| IMAGE_ENCODE_WRITE (PNG) | 0.366 s | 0.359 s | |
| TOTAL_PROCESS | 10.758 s | 17.090 s | |

**Inference-only (denoise) ≈ 2.92 s/step × 28 ≈ 81.8 s.**
Model load ≈ 5.7–6.3 s is a separate, non-inference cost.

---

## 6. Actual 2048 sequence dimensions

| Quantity | Value |
|---|---|
| Grid | 64 × 64 |
| image_len (generation tokens) | 4096 |
| text_len (AR / prompt tokens) | 19 |
| **Total sequence length S** | **4115** |
| Hidden H | 4096 |
| Heads NH / NKV | 32 / 8 (GQA 4:1) |
| Head dim HD | 128 |
| FF hidden | 12288 |
| Head out | 3072 |
| Layers | 36 |
| Patch | 32, fix_point 4096, eps 1e-6, rope theta 5e6, mrope [24,20,20] |

---

## 7. B0 — GEMM microbench (real 2048 shapes)

`tests/unit/bench_gemm.c`, S=4115, 50 iters, bf16→bf16, cuBLAS.

| GEMM | M | N | K | ms/call | TFLOP/s |
|---|---|---|---|---|---|
| q_proj | 4115 | 4096 | 4096 | 4.775 | 28.92 |
| k_proj | 4115 | 1024 | 4096 | 1.210 | 28.53 |
| v_proj | 4115 | 1024 | 4096 | 1.195 | 28.89 |
| o_proj | 4115 | 4096 | 4096 | 4.811 | 28.70 |
| gate_proj | 4115 | 12288 | 4096 | 13.893 | 29.81 |
| up_proj | 4115 | 12288 | 4096 | 13.962 | 29.67 |
| down_proj | 4115 | 4096 | 12288 | 13.774 | 30.07 |
| final_head | 4115 | 3072 | 4096 | 3.595 | 28.80 |
| te0 (M=1) | 1 | 4096 | 256 | 0.033 | — |
| te2 (M=1) | 1 | 4096 | 4096 | 0.222 | — |

All shapes are **identical across all 36 blocks and all 28 steps**.
Achieved ≈ 29–30 TFLOP/s uniformly (peak bf16 not independently measured).

---

## 8. B1/B2/B3/B4 — block / transformer / step profiles

Source: `build/hidream_timing_block`, 3-step run (108 block invocations).
Per-block = total / 108. Per-transformer = per-block × 36.

| Stage | ms/block | ms/transformer | % of transformer GPU |
|---|---|---|---|
| B_mlp_gate_up | 27.337 | 984.1 | **34.7 %** |
| B_mlp_down | 13.589 | 489.2 | **17.3 %** |
| B_sdpa | 13.550 | 487.8 | **17.2 %** |
| B_qkv_proj | 7.136 | 256.9 | 9.1 % |
| B_o_proj | 4.735 | 170.5 | 6.0 % |
| B_swiglu | 1.494 | 53.8 | 1.9 % |
| B_qk_norm | 1.472 | 53.0 | 1.9 % |
| B_head_split | 0.952 | 34.3 | 1.2 % |
| B_rope | 0.935 | 33.7 | 1.2 % |
| B_attn_resid | 0.485 | 17.5 | 0.6 % |
| B_mlp_resid | 0.475 | 17.1 | 0.6 % |
| B_input_norm | 0.322 | 11.6 | 0.4 % |
| B_post_norm | 0.309 | 11.1 | 0.4 % |
| B_final_copy | 0.281 | 10.1 | 0.4 % |
| B_mrope | 0.048 | 1.7 | 0.06 % |
| **BLOCK_SINGLE** | **73.145** | **2633.3** | 92.9 % |

Transformer-level regions (per forward):

| Region | ms/forward | % of transformer GPU |
|---|---|---|
| EMBEDDING | 197.6 | 7.0 % |
| BLOCKS_TOTAL | 2633.3 | 92.9 % |
| FINAL_NORM_HEAD | 4.5 | 0.16 % |
| **TRANSFORMER_TOTAL** | **2835.4** | 100 % |
| SCHEDULER (outside transformer) | 2.3 | — |

**B3 (1 step):** DENOISE_TOTAL 3.460 s CPU; transformer GPU 3.218 s.
**B4 (3 steps):** DENOISE_TOTAL 8.767 s CPU (2.92 s/step); transformer GPU
8.506 s (2.835 s/forward). Non-GPU denoise overhead ≈ 0.085 s/step
(host RNG over 12.58 M floats + clip + 50 MB H2D noise copy + launch gaps).

### EMBEDDING sub-regions (per forward)

| Region | ms/forward |
|---|---|
| E_timestep | 193.1 |
| E_xembed | 3.0 |
| E_gather_tms | 1.2 |
| E_cat | 0.3 |

**E_timestep is a one-time cost, not per-step.** 1-step EMBEDDING = 0.5142 s;
3-step EMBEDDING = 0.5929 s → first forward ≈ 0.514 s, subsequent ≈ 0.039 s
each. The 0.193 s/forward figure is an artifact of dividing a one-time
cuBLAS GEMV JIT/attribute cost across 3 forwards. Over 28 steps it costs
≈ 0.5 s total (~0.6 %), not ~5 %.

---

## 9. B5 — full CLI generation

See §5. **84.1–86.0 s** end-to-end, of which ≈ 81.8 s is denoise compute and
≈ 5.7–6.3 s is model load.

---

## 10. GEMM inspection

Semantics (all production GEMMs):

```
y[M,N] = x[M,K] · w[N,K]^T
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
             ..., A=w ld=K, B=x ld=K, C=y ld=N,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
```

* input dtype bf16, output dtype bf16, compute dtype fp32.
* bias is added by a **separate** `hd_gemm_bias_add_kernel` (not fused).
* backend: plain `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT` — **no cuBLASLt
  path in the hot path**. `hd_gemm_runtime_init` is called with
  `workspace_bytes = 0`; the cuBLASLt handle exists but is idle.
* calls per block: 7 (q, k, v, o, gate, up, down).
* calls per transformer: 252 + 1 final head.
* **All shapes repeat identically across blocks and steps.**

---

## 11. SDPA inspection

| Field | Value |
|---|---|
| Q shape | [1, 32, 4115, 128] bf16 |
| K/V shape | [1, 8, 4115, 128] bf16 |
| Heads / KV heads | 32 / 8 (GQA 4:1) |
| Head dim | 128 |
| AR length | 19 |
| Generation length | 4096 |
| Mask mode | additive bias [1,1,S,S] bf16 |
| Workspace | ≈ 3.06 GB, allocated once in `hd_sdpa_create` |
| Time per call | 13.13 ms (nsys) / 13.55 ms (block timing) |
| Time per block | 13.55 ms |
| Time per transformer | 487.8 ms (36 calls) |
| Plan reused | **yes** — graph + execution plan built once per generation |
| Workspace reused | **yes** |
| Mask materialization | **yes, once per generation** — host loop over S²=16.9 M entries + 33.9 MB H2D |

Kernel: `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x128`
— 36 instances (1/block), **no eager fallback** at S=4115.

---

## 12. Hot-path lifecycle audit

From `nsys` on a 1-step 2048 generation.

### Per-step / per-block (hot path)

| Event | Count | Cost | Verdict |
|---|---|---|---|
| `cudaLaunchKernel` | 523 | 33.7 ms | expected |
| `cudaMemcpyAsync` (D2D) | 42 | 3.58 ms / 1325 MB | expected (vinput cat, z copies) |
| `cudaMemcpy` (H2D noise) | 1/step | 50 MB | expected |
| `cudaStreamSynchronize` | 1 | 1.69 ms | expected |
| `cudaDeviceSynchronize` | 5 | 33.5 µs | expected (final) |
| `cudaMemsetAsync` | 2 | — | expected |
| `cudaMalloc` / `cudaFree` in loop | **0** | — | **clean** |
| host alloc in loop | 0 | — | **clean** |
| cuBLAS handle creation in loop | 0 | — | **clean** |
| cuDNN plan construction in loop | 0 | — | **clean** (built once) |
| workspace alloc in loop | 0 | — | **clean** |
| weight rebinding in loop | 0 | — | **clean** |

### One-time (startup / first call)

| Event | Count | Cost | Reason |
|---|---|---|---|
| `cudaMalloc` | 25 | 3.145 s (one call 2.56 s) | weight + workspace alloc |
| `cudaMallocHost` | 4 | 0.410 s | pinned staging |
| `cuKernelSetAttribute` | 75 | 2.643 s | cuBLAS GEMV JIT (first call) |
| `cuLibraryLoadData` | 22 | 0.868 s | cuBLAS kernel library load |
| H2D weight load | 132 | 987 ms / 17.74 GB | model load (7.43 GB/s disk) |
| D2H final z | 1 | 7.50 ms / 25.2 MB | output reconstruction |

**No unexpected per-step allocations, frees, syncs, handle creations, plan
constructions, or weight rebinds were observed in the hot path.**

### Latent anomaly (NOT on the production path)

`src/cuda/attn.cu` `hd_attention_eager()` performs a per-call
`cudaMalloc` (2.17 GB fp32 scratch) + `cudaDeviceSynchronize()` + `cudaFree`.
It is the **silent fallback** in `hd_decoder_block` (`if (rc != 0)
hd_attention_eager(...)`). It is not exercised at 2048 because cuDNN SDPA
succeeds, but any future SDPA failure would silently activate it.

---

## 13. Bottlenecks (sorted by measured time)

Per denoise step, GPU ≈ 2.838 s (transformer 2.835 s + scheduler 0.002 s).

| Component | ms/step | % transformer GPU | % denoise-step GPU |
|---|---|---|---|
| MLP gate+up GEMMs | 984.1 | 34.7 % | 34.7 % |
| MLP down GEMM | 489.2 | 17.3 % | 17.2 % |
| SDPA | 487.8 | 17.2 % | 17.2 % |
| QKV GEMMs | 256.9 | 9.1 % | 9.1 % |
| O GEMM | 170.5 | 6.0 % | 6.0 % |
| EMBEDDING (one-time) | ~18 (amortized) | 0.6 % | 0.6 % |
| SwiGLU | 53.8 | 1.9 % | 1.9 % |
| qk_norm | 53.0 | 1.9 % | 1.9 % |
| head_split | 34.3 | 1.2 % | 1.2 % |
| RoPE | 33.7 | 1.2 % | 1.2 % |
| residual adds | 34.6 | 1.2 % | 1.2 % |
| input/post norm | 22.7 | 0.8 % | 0.8 % |
| final_copy | 10.1 | 0.4 % | 0.4 % |
| mrope | 1.7 | 0.06 % | 0.06 % |
| final norm+head | 4.5 | 0.16 % | 0.16 % |
| scheduler | 2.3 | — | 0.08 % |
| non-GPU denoise overhead | 85 (CPU) | — | 3.0 % |

**GEMMs dominate (67.1 % of transformer GPU time).** SDPA is 17.2 %.
Pointwise/norm/RoPE kernels together are ≈ 7.5 %.

---

## 14. Correctness

| Test | Result |
|---|---|
| `test-primitives` | 33 assertions passed, 0 failed |
| `test-block` | 7 assertions passed, 0 failed |
| `test-sdpa-block` | 4 passed, 0 failed (sdpa 107.7× faster than eager) |
| `test-sdpa-forward` | 4 passed, 0 failed (sdpa 6.52× faster than eager) |
| `test-full-forward` | 13 passed, **1 failed** |

The single failure is the **known pre-existing** `complete_output`
(`nrmse=0.15267 cos=0.98882141`, `tests/unit/test_full_forward.c:173`).
It reproduces the baseline value exactly. **Threshold unchanged.**

Instrumentation did not alter model output: reference GEMM backend,
production cuBLAS backend, numerical thresholds, vision/reference path and
scheduler behavior are all untouched.

---

## 15. Instrumentation diff

`artifacts/m2/perf/instrumentation.diff` (297 lines). Summary:

* `src/runtime/o1_timing.c` — region cap 64→128, event cap 4096→65536.
* `src/runtime/o1_timing.h` — `O1_BTIMING_BEGIN_GPU/END_GPU` macros gated by
  `-DO1_DEBUG_BLOCK_TIMING` + no-op fallbacks.
* `src/model/block.c` — 15 GPU regions across the 12 block stages.
* `src/model/forward.c` — 4 EMBEDDING sub-regions.
* `src/main.c` — report path honors `O1_TIMING_JSON`.
* `Makefile` — `timing-block` and `bench-gemm` targets.
* `tests/unit/bench_gemm.c` — new B0 microbench.

---

## 16. Candidate optimizations (analysis only — none implemented)

| # | Candidate | Evidence | Affected runtime | Max benefit | Complexity | Correctness risk | Validation microbench |
|---|---|---|---|---|---|---|---|
| C1 | cuBLASLt with cached algorithms | 67 % of transformer is GEMM; all shapes repeat identically; current path uses `CUBLAS_GEMM_DEFAULT` with no workspace | GEMMs | up to ~10–20 % of GEMM time | medium | low | B0 re-run with cuBLASLt + cached algo |
| C2 | Fuse bias into GEMM epilogue | separate `hd_gemm_bias_add_kernel` per GEMM | GEMM epilogues | small (~1 %) | low | low | B0 with epilogue |
| C3 | cuDNN SDPA tuning (heuristic mode / knob) | SDPA 17.2 %; plan built with `HeurMode_t::A` | SDPA | up to ~10–20 % of SDPA | low | low | B1 SDPA-only |
| C4 | Two-pass / FlashAttention-style attention | SDPA 17.2 %; 3.06 GB workspace | SDPA | up to ~17 % of transformer | high | medium | B1 SDPA-only |
| C5 | CUDA Graph capture of the denoise step | 523 launches/step, 33.7 ms launch API; 0 per-step allocs | launch overhead | up to ~1–3 % | medium | medium | B4 with/without graph |
| C6 | Pointwise fusion (norm+RoPE+residual) | ~7.5 % in small kernels | pointwise | up to ~5 % | medium | medium | B1 sub-stage |
| C7 | Workspace/allocation cleanup | already clean in hot path | — | ~0 | low | low | nsys re-audit |
| C8 | Scheduler device migration | 0.085 s/step non-GPU overhead (3 %) | denoise host overhead | up to ~3 % | medium | low | B4 |

No winner selected — pending review.

---

## 17. Artifacts

```
artifacts/m2/perf/
  run_baseline.sh
  baseline/{warmup,run1,run2,run3}.{png,json,log,time}
  b0_gemm.txt
  b1_block_1step.{json,png}
  b4_3step.{json,png}
  nsys_1step.nsys-rep, nsys_1step.sqlite
  instrumentation.diff
```