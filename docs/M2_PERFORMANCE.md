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

---

# Part 2 — cuBLASLt production GEMM path

Commits:

| Purpose | Hash | Contents |
|---|---|---|
| instrumentation baseline | `6956cf0` | `o1_timing.{c,h}`, `block.c`, `forward.c`, `main.c`, `Makefile`, `bench_gemm.c`, this ledger |
| cuBLASLt optimization | `4f09036` | `src/cuda/gemm.cu`, `src/cuda/gemm.h`, `src/model/weights.c`, `tests/unit/bench_gemm.c` |

## 18. Audit of pre-existing cuBLASLt work

`git log -S'cublasLt' --all --oneline` returns no commit: there was **no
committed cuBLASLt implementation** in history. What existed was an
*uncommitted working-tree* implementation in `src/cuda/gemm.cu` (a
`hd_lt_plan` struct, `hd_lt_run`, `hd_lt_tune`, `hd_gemm_cublaslt`) that had
never been wired into the production dispatch. `git grep -n cublasLt` before
this task matched only that uncommitted diff; `HD_GEMM_CUBLASLT` does not
exist anywhere in the tree.

**Root cause of the plain-`cublasGemmEx` profile.** Three independent
reasons, all verified in code:

1. `hd_linear()` dispatched on `g_prod_backend`, whose default is `0`, but
   the `hd_gemm_cublaslt` branch was only reachable when `g_rt->lt` was
   non-NULL *and* a plan could be built. The plan cache did not exist, so
   every call fell through to `hd_gemmex_run`.
2. `hd_gemm_runtime_init()` was called from `src/model/weights.c` as
   `hd_gemm_runtime_init(device_id, 0)` — a **zero-byte workspace**, so any
   candidate algorithm requiring workspace was rejected and the Lt path had
   no persistent buffer to run in.
3. Nothing selected a *tuned* algorithm: there was no cached algorithm, so
   even a successful Lt call would have used an unselected default.

So the answer to "why were we profiling plain cuBLAS" is **not** a
regression or a revert: the intended Lt path had simply never been completed
and committed. `cublasGemmEx` was doing exactly what the code said.

## 19. Exact unique 2048 GEMM shapes

All shapes are identical across all 36 blocks and all 28 denoise steps.
Measured `S = 4122` for the 3-step probe (the frozen 28-step run reports
`S = 4115`; the difference is the CFG text/generation split and does not
change the shape set).

| GEMM | M | N | K | dtype | compute | calls/block | calls/transformer |
|---|---|---|---|---|---|---|---|
| q_proj | 4122 | 4096 | 4096 | bf16→bf16 | fp32 | 1 | 36 |
| k_proj | 4122 | 1024 | 4096 | bf16→bf16 | fp32 | 1 | 36 |
| v_proj | 4122 | 1024 | 4096 | bf16→bf16 | fp32 | 1 | 36 |
| o_proj | 4122 | 4096 | 4096 | bf16→bf16 | fp32 | 1 | 36 |
| gate_proj | 4122 | 12288 | 4096 | bf16→bf16 | fp32 | 1 | 36 |
| up_proj | 4122 | 12288 | 4096 | bf16→bf16 | fp32 | 1 | 36 |
| down_proj | 4122 | 4096 | 12288 | bf16→bf16 | fp32 | 1 | 36 |
| final head | 4122 | 3072 | 4096 | bf16→bf16 | fp32 | — | 1 |
| tms embed | 1 | 4096 | 256 | bf16→bf16 | fp32 | — | 1 |
| tms proj | 1 | 4096 | 4096 | bf16→bf16 | fp32 | — | 1 |

Layout contract: `Y[M,N] = X[M,K] · W[N,K]^T`, row-major X/W/Y.
cuBLASLt descriptors: `Adesc` = X `[M,K]` ld=K `ORDER_ROW`; `Bdesc` = W
`[N,K]` ld=K `ORDER_ROW` with `TRANSA=N`, `TRANSB=T`; `Cdesc` = Y `[M,N]`
ld=N `ORDER_ROW`. Bias is still applied by the separate
`hd_gemm_bias_add_kernel`.

## 20. Implementation changes

- `hd_lt_plan` cache (32 slots) keyed on `(M,N,K)`; descriptors built once
  per shape, reused for all blocks and steps.
- `hd_lt_tune()` — enumerates the heuristic candidates, rejects any that
  does not reproduce the `cublasGemmEx` oracle bit-for-bit, benchmarks the
  survivors with CUDA events (3 iterations, best-of) and caches the fastest.
  Runs once per unique shape, never in steady state.
- `hd_gemm_cublaslt()` — `O1_GEMM_TUNE=0` uses `heuristic[0]`; `tune=1`
  (default) uses the tuner. Added an explicit `cublasGemmEx` fallback when no
  algorithm was selected, instead of launching an unselected algorithm.
- `hd_gemm_set_prod_backend()` added to `gemm.h`; `O1_GEMM_BACKEND=ex`
  forces the GemmEx sub-backend for A/B.
- `weights.c`: `hd_gemm_runtime_init(device_id, 0)` → `(device_id, 64u<<20)`
  in both `hd_weights_to_device` and `hd_weights_to_device_gguf`.
- Env knobs: `O1_GEMM_TUNE`, `O1_GEMM_DEBUG`, `O1_GEMM_BACKEND`.

## 21. Selected algorithm and workspace per shape

`O1_GEMM_DEBUG=1`, 3-step probe:

| Shape (M×N×K) | candidates | selected | workspace | tuned ms |
|---|---|---|---|---|
| 1×4096×256 | 9 | best=4 | 0 | 0.015 |
| 1×4096×4096 | 8 | best=0 | 0 | 0.222 |
| 4122×4096×4096 (q,o) | 3 | best=0 | 0 | 4.662 |
| 4122×1024×4096 (k,v) | 6 | best=0 | 0 | 1.195 |
| 4122×12288×4096 (gate,up) | 3 | best=0 | 0 | 13.576 |
| 4122×4096×12288 (down) | 3 | best=0 | 0 | 14.152 |
| 4122×3072×4096 (head) | 3 | best=1 | 0 | 3.449 |

**Every selected algorithm requires 0 bytes of workspace** — the 64 MiB
persistent buffer is allocated but unused by these selections. It is kept as
headroom so that a future shape or cuBLAS version can select a
workspace-backed algorithm without a hot-path allocation.

The algorithm space on this platform is very small: the heuristic returns
3 candidates for the large shapes, and `cublasLtMatmulAlgoGetIds` returns 20
IDs of which only 1 passes `AlgoCheck`. `CUBLAS_COMPUTE_16F` with BF16
inputs is `NOT_SUPPORTED` (status 15), so FP32 accumulate is mandatory.

## 22. Device ceiling

| GEMM | ms | TFLOP/s |
|---|---|---|
| 2048³ | 0.644 | 26.67 |
| 4096³ | 4.636 | 29.65 |
| 8192³ | 35.523 | 30.95 |
| 12288³ | 117.197 | 31.66 |

Measured device ceiling ≈ **31.7 TFLOP/s** (bf16 in / fp32 acc, GB10, 48 SM).
The production shapes run at 29–31 TFLOP/s, i.e. **~94 % of the achievable
ceiling**. fp16 gives no headroom (32.00 TFLOP/s fp16/fp32acc, 31.51
fp16/fp16acc — measured, same ceiling).

## 23. GEMM before/after — B0 microbench (real 2048 shapes)

`./build/bench_gemm 4115 50`:

| GEMM | M | N | K | Lt ms | Lt TF | GemmEx ms | GemmEx TF | ratio |
|---|---|---|---|---|---|---|---|---|
| q_proj | 4115 | 4096 | 4096 | 4.614 | 29.93 | 4.643 | 29.74 | 1.01× |
| k_proj | 4115 | 1024 | 4096 | 1.153 | 29.94 | 1.176 | 29.35 | 1.02× |
| v_proj | 4115 | 1024 | 4096 | 1.158 | 29.80 | 1.177 | 29.34 | 1.02× |
| o_proj | 4115 | 4096 | 4096 | 4.618 | 29.90 | 4.612 | 29.94 | 1.00× |
| gate_proj | 4115 | 12288 | 4096 | 13.538 | 30.60 | 13.571 | 30.52 | 1.00× |
| up_proj | 4115 | 12288 | 4096 | 13.560 | 30.55 | 13.567 | 30.53 | 1.00× |
| down_proj | 4115 | 4096 | 12288 | 13.555 | 30.56 | 13.595 | 30.47 | 1.00× |
| final_head | 4115 | 3072 | 4096 | 3.483 | 29.73 | 3.516 | 29.45 | 1.01× |
| te0 (M=1) | 1 | 4096 | 256 | 0.014 | 0.15 | 0.033 | 0.06 | 2.29× |
| te2 (M=1) | 1 | 4096 | 4096 | 0.209 | 0.16 | 0.217 | 0.15 | 1.04× |

**cuBLASLt and cublasGemmEx are within 1 % on every dominant production
shape.** The only material win is the tiny `M=1` timestep GEMM (2.29×), worth
~0.02 ms/step. There is no GEMM headroom left to recover.

## 24. Block / transformer / step before/after

3 denoise steps = 108 blocks. GPU seconds from the block timing build.

| region | GemmEx (before) | Lt tuned (after) | Δ |
|---|---|---|---|
| B_mlp_gate_up | 3.0395 | 3.5636 | one-time tuning |
| B_mlp_down | 1.5041 | 1.8461 | one-time tuning |
| B_qkv_proj | 0.7881 | 1.0966 | one-time tuning |
| B_o_proj | 0.5230 | 0.5197 | — |
| B_sdpa | 1.4994 | 1.5047 | — |
| BLOCK_SINGLE | 8.0974 | 9.2750 | — |
| TRANSFORMER_TOTAL | 8.6760 | 10.1308 | — |
| per block (ms) | 74.98 | 85.88 | — |

The delta is **entirely the one-time first-touch tuning cost**, which is
charged to whichever block first executes each unique shape. Steady state is
identical: with the tuner warm, `tune=0` (heuristic[0]) measures 74.10 ms/block
and `tune=1` 74.10 ms/block — both within noise of the GemmEx 74.98 ms/block
baseline.

Steady-state per-block profile with the cuBLASLt backend (tuned), 3 steps:

| region | GPU s | count | ms/block |
|---|---|---|---|
| B_input_norm | 0.0348 | 108 | 0.322 |
| B_qkv_proj | 0.7906 | 108 | 7.320 |
| B_head_split | 0.1052 | 108 | 0.974 |
| B_qk_norm | 0.1627 | 108 | 1.506 |
| B_mrope | 0.0052 | 108 | 0.048 |
| B_rope | 0.1032 | 108 | 0.956 |
| B_sdpa | 1.4995 | 108 | 13.884 |
| B_o_proj | 0.5228 | 108 | 4.840 |
| B_attn_resid | 0.0526 | 108 | 0.487 |
| B_post_norm | 0.0330 | 108 | 0.305 |
| B_mlp_gate_up | 3.0405 | 108 | 28.152 |
| B_swiglu | 0.1654 | 108 | 1.532 |
| B_mlp_down | 1.5087 | 108 | 13.969 |
| B_mlp_resid | 0.0516 | 108 | 0.478 |
| B_final_copy | 0.0306 | 108 | 0.283 |
| **BLOCK_SINGLE** | **8.1092** | 108 | **75.085** |
| EMBEDDING | 0.5090 | 3 | 4.713 |
| TRANSFORMER_TOTAL | 8.6327 | 3 | 79.933 |
| FINAL_NORM_HEAD | 0.0144 | 3 | 0.133 |
| SCHEDULER | 0.0070 | 3 | 0.065 |

## 25. Full CLI 28-step before/after

| run | GemmEx baseline (s) | Lt tuned (s) |
|---|---|---|
| warmup | 85.55 | 84.55 |
| run1 | 86.00 | 85.74 |
| run2 | 84.11 | 86.18 |
| run3 | 84.08 | 85.22 |
| max RSS | 823 MB | 887 MB |

Identical within run-to-run noise (±1.5 s). The cuBLASLt path is a
**correctness/lifecycle completion, not a speedup** — as the ~94 %-of-ceiling
measurement predicts.

## 26. Numerical results

| comparison | max_abs | mean_abs | NRMSE | cosine | pixels differing |
|---|---|---|---|---|---|
| Lt tuned vs GemmEx, 3-step | 0 | 0 | 0 | 1.000000 | 0 % |
| Lt tuned vs GemmEx, 28-step | 0 | 0 | 0 | 1.000000 | 0 % |
| Lt heuristic[0] vs GemmEx, 3-step | 229 | 51.0 | 0.2455 | 0.877944 | 99.4 % |
| reference GEMM vs GemmEx | — | — | 0.1189 | 0.9929 | 87.9 % |

Key result: the **tuned** cuBLASLt selection reproduces `cublasGemmEx`
**bit-for-bit** (identical PNG md5 `60614703addca835ac80c3d6858ecd88` at
3 steps and `f0932876085fc9255117fae49e43d2a8` at 28 steps). The raw
`heuristic[0]` candidate does **not** (cos 0.878) — which is exactly what the
tuner's bit-exact filter exists to prevent. Tuning therefore stays the
default: it costs a one-time ~1.1 s but buys bit-exactness at identical
steady-state throughput.

The residual ~0.99 divergence between the reference kernel and the cuBLAS
family is the previously-accepted pre-existing difference and is unchanged by
this work.

## 27. Correctness ladder

| test | result |
|---|---|
| `test_gemm_smoke` | PASS (2×2, 4×3×5) |
| `test_primitives` | 33 assertions, 0 failed |
| `test_block` | 7 assertions, 0 failed |
| `test_sdpa_block` | 4 assertions, 0 failed |
| `test_sdpa_forward` | 4 assertions, 0 failed |
| `test_layer_replay` | REPLAY PASS |
| `test_full_forward` | 13 passed, 1 failed |

`test_full_forward`'s `complete_output` reports
`nrmse=0.15267 cos=0.98882141` — the **known pre-existing** failure, unchanged
and with its threshold untouched. No new regression.

## 28. Hot-path lifecycle (cuBLASLt path)

No `cublasLtCreate`, no descriptor creation, no heuristic search, no
`cudaMalloc`/`cudaFree`, no workspace allocation and no device synchronization
occur in steady state. All of the above happen once per unique shape during
the first block that touches it. `hd_lt_run` re-runs the selected algorithm
after tuning, so the first call's output is guaranteed to come from the
cached algorithm.

## 29. Conclusion and next target

The GEMM workload runs at ~94 % of the measured device ceiling. Completing
and tuning the cuBLASLt path made the backend correct, deterministic and
bit-exact against `cublasGemmEx`, with a clean persistent lifecycle, but it
does **not** and cannot yield a meaningful speedup on this hardware.

**C1 (cuBLASLt with cached algorithms) is therefore closed as complete and
non-beneficial.** The remaining measured headroom is elsewhere: SDPA 17.2 %,
MLP pointwise (SwiGLU 1.5 ms/block, norms 2.1 ms/block), and the ~3 %
non-GPU denoise overhead. The next optimization target should be selected
from C3/C4/C6, not from GEMM.

## 30. Artifacts (Part 2)

```
artifacts/m2/perf/
  b0_gemm_lt.txt
  lt_layout_probe.txt, lt_peak_probe.txt, lt_compute_type_probe.txt,
  lt_algo_ids_probe.txt, lt_fp16_ceiling_probe.txt
  after/{warmup,run1,run2,run3}.{png,json,log,time}      (Lt tuned)
  final/{b4_3step.*, cmp_default.*, cmp_tune1.*, cmp_ex.*}
  final2/{t1.*, ex.*}                                    (28-step A/B)
  run_after.sh
```
```
---

# Part 3 — C4: two-pass SDPA

## 31. Motivation and scope

Steady state (Part 2, §24) showed `B_sdpa = 13.884 ms/block` ≈ 17.2 % of
transformer GPU time, on the single mixed-mask cuDNN graph. The t2i mask is
*exactly* "causal for rows `[0, text_len-1)`, fully unmasked from
`text_len-1` onward", so the single graph pays for a mask/bias path that is
only needed on **18 of 4115 rows**.

Scope was deliberately limited to this one target: no SDPA variants, no CUDA
Graph, no fusion, no GEMM work (C1 is closed).

## 32. Decomposition

```
ar_len = text_len - 1 = 18      (text_len = 19)
gen_len = S - ar_len  = 4097

PASS 1  Q[0:ar_len]   K[0:ar_len]   V[0:ar_len]   causal=true,  no bias
PASS 2  Q[0:S]        K[0:S]        V[0:S]        causal=false, no bias
```

**Key structural insight.** Because the mask zeroes *every* row
`r >= ar_len` across *all* columns, pass 2 is not a masked subset — it is
**exactly plain full non-causal attention of all S queries over all S keys**.
Pass 2 therefore needs no mask, no bias, and **no staging or striding**: it
runs directly on the real contiguous `[H,S,D]` buffers. Only the 18-row
causal head needs compact staging.

## 33. Implementation

`src/cuda/hd_cudnn_sdpa.cu`

* `struct hd_sdpa_pass { graph, q, k, v, o, workspace, workspace_bytes }`.
* `hd_sdpa_plan` gained `int ar_len`, `hd_sdpa_pass ar, gen`, and one-time
  staging buffers `q_stage/k_stage/v_stage/o_stage` + `stage_bytes`
  (`cudaMalloc`ed in `hd_sdpa_create_split`, freed in the destructor).
* `hd_sdpa_build_pass(...)` — **packed strides** `{H*S*D, S*D, D, 1}`.
* `hd_sdpa_execute_split` — `cudaMemcpy2DAsync` gather (src pitch `Sq*D*2`,
  dst pitch `ar_len*D*2`, height = heads) → **pass 2 first** (writes the whole
  output on the real buffers) → pass 1 on staging → `cudaMemcpy2DAsync`
  scatter of `o_stage` back into `out`.
* `hd_sdpa_execute` auto-dispatches to the split when `plan->ar_len > 0`.
* `hd_sdpa_create_prod` (`extern "C"`) — production entry point; split when
  `0 < ar_len < seq`, else the masked plan (fallback preserved).

`src/runtime/generate.c` — the t2i plan site now calls
`hd_sdpa_create_prod(&plan, 1, NH, NKV, Ss, ar_len, HD, attn_scale)` with
`ar_len = seq_text_len[b] - 1`. The vision site (`generate.c:549`, head_dim
72, different mask) and the generic `hd_seq_build` mask path are untouched.

**Rejected alternative.** Expressing the AR slice as a *non-compact strided
view* (head stride > `seq*D`) passes `validate`/`check_support`/`build_plans`
but **fails at launch** with
`err 716 CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER` (`shimCuLaunchKernelEx`).
This is **not** causal-specific — reproduced with `causal=false`. cuDNN
rejects non-compact head strides, so compact staging is the working route.

## 34. Correctness

Microbenchmark (`build/bench_sdpa_twopass`, real production geometry
GRID 64x64, TEXT_LEN 19, SEQLEN 4115, NH 32, NKV 8, HD 128), mixed-mask vs
two-pass over all 32 heads:

```
max_abs = 0
NRMSE   = 0
cosine  = 1.000000000
head0 causal-head NRMSE=0 cos=1 | full-tail NRMSE=0 cos=1
```

Bit-exact. Existing regression tests, unchanged thresholds:

```
test-sdpa-block    eager vs sdpa: nrmse=0 cos=1  -> PASS
test-sdpa-forward  eager vs sdpa: nrmse=0 cos=1  -> PASS
test-full-forward  complete_output nrmse=0.15267 cos=0.98882141 -> FAIL
```

`complete_output` is the **known pre-existing** failure at exactly the
baseline value (`0.15267 / 0.98882141`), threshold untouched: no new
regression.

End-to-end: the canonical 28-step PNG is **bit-identical** to the masked
28-step PNG (`sha256 c4e1e2e8b1f08d7c…`, 0 differing bytes of 12 584 960).

## 35. SDPA A/B (CUDA events, real production dims)

```
current mixed-mask SDPA : 14.0002 ms
two-pass pass1 (causal) :  0.0448 ms
two-pass pass2 (full)   : 10.6721 ms
two-pass total          : 10.6427 ms
speedup                 : 1.3155x  (23.98% faster)
mixed effective         : 19.82 TFLOP/s
two-pass effective      : 26.07 TFLOP/s
```

Plan build: mixed 0.091 s, two-pass 0.098 s (once per generation).
Steady state contains no plan build, no `cudaMalloc`, no synchronization;
the two `cudaMemcpy2DAsync` gathers/scatters are async on the same stream.

## 36. Block / transformer / step before/after

All rows below are **controlled A/B runs on the same machine state with
identical GEMM code** (only `ar_len` toggled), because wall-clock and clock
drift made the historical 10:36 baseline non-comparable.

**B1 — one block** (per-block GPU seconds, 36 blocks)

| region | masked | two-pass | delta |
|---|---|---|---|
| `B_sdpa` | 13.781 ms | 11.436 ms | **-17.01 %** |
| `BLOCK_SINGLE` | 104.3 ms | 103.0 ms | -1.19 % |

**B2/B3 — one denoise step** (`TRANSFORMER_TOTAL`, 1 step)

| region | masked | two-pass | delta |
|---|---|---|---|
| `B_sdpa` | 13.781 ms | 11.436 ms | -17.01 % |
| `BLOCKS_TOTAL` | 3.7542 s | 3.7096 s | -1.19 % |
| `TRANSFORMER_TOTAL` | 4.5390 s | 4.4685 s | -1.55 % |

**B4 — three denoise steps**

| region | masked | two-pass | delta |
|---|---|---|---|
| `B_sdpa` / block | 13.6 ms | 11.4 ms | **-15.80 %** |
| `BLOCK_SINGLE` | 82.6 ms | 81.4 ms | -1.41 % |
| `BLOCKS_TOTAL` | 8.9166 s | 8.7912 s | -1.41 % |
| `TRANSFORMER_TOTAL` | 9.7415 s | 9.6183 s | -1.27 % |

The SDPA gain survives at block/transformer level with the expected
amortisation: SDPA is ~17 % of transformer time, so a ~16 % SDPA cut is
~1.3–1.5 % of the transformer — exactly what is observed. No anomalous
launch/copy/sync overhead was found; the two extra `cudaMemcpy2DAsync`
gather/scatter plus one extra graph launch account for the residual.

**B5 — full CLI, 28 steps** (Dev, 2048x2048, seed 42, same prompt/scheduler)

| run | wall clock |
|---|---|
| masked (controlled) | 85.42 s |
| two-pass | 83.70 s |

Delta **-1.72 s (-2.0 %)**. Historical baseline (10:36) was 84.08–86.00 s;
the machine drifted upward afterwards, hence the matched masked control.

## 37. Decision

```
correctness gate (nrmse<=1e-2 && cos>=0.999) : PASS  (nrmse=0, cos=1)
performance gate (>=15% faster)              : PASS  (-17.0 % / -23.98 %)
=> ACCEPTED
```

C4 is integrated as the production SDPA path for the t2i sequence layout.
The masked single-graph path remains available as fallback and for the
vision / generic-mask call sites.

## 38. Artifacts (Part 3)

```
artifacts/m2/perf/c4/
  b3_masked.json/.png      b3_1step.json/.png     (1-step controlled A/B)
  b4_masked.json/.png      b4_twopass.json/.png   (3-step controlled A/B)
  b5_masked.{png,time}     b5_28step.{png,time}   (28-step controlled A/B)
  b4_3step.{json,log,png}  iso.png
tests/unit/bench_sdpa_twopass.cu                  (make bench-sdpa-twopass)
```

## 39. Next target

SDPA is now ~11.4 ms/block. Remaining measured headroom: GEMM (already at
~94 % of device ceiling, closed), MLP pointwise (SwiGLU ~1.5 ms/block, norms
~2.1 ms/block), and the ~3 % non-GPU denoise overhead. Selection is out of
scope for C4.

---

# Part 4 — Native EDIT path

Scope: the **native CLI edit path** (`hidream_timing_block --mode edit`), not the
server. Canonical workload:

```
Dev edit / 2048x2048 target / 1x 2048x2048 reference / 28 steps / FlowMatch
seed 42 / prompt "Place the person in a snowy mountain landscape"
```

## 40. Canonical command

```
export LD_LIBRARY_PATH=/home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
O1_TIMING_JSON=artifacts/m2/edit/final.json \
./build/hidream_timing_block --mode edit \
  --ref-image _reference2/HiDream-O1-Image/assets/edit/test.jpg \
  --width 2048 --height 2048 --steps 28 --seed 42 \
  --prompt "Place the person in a snowy mountain landscape" \
  --output artifacts/m2/edit/final.png
```

Output SHA-256 `21ca624be8e41dbea30d2c4128ad1eeaa11fd99a6d067a9f544d77fe57a58464`
(reproducible run-to-run).

## 41. Edit geometry (measured, one canonical run)

```
target_img_tokens    4096      (64x64 patch grid)
ref_pixel_tokens     4032
total_img            8128
text_len              150
S                    8278
vlm_N (raw)           520
vlm_V (merged)        130
forward_workspace     10 559 525 376 B
decoder_scratch       10 220 747 520 B
mask                  137 050 568 B
```

Attention backends:

```
main decoder : CUDNN_SDPA   1008/1008 calls  (0 eager fallback)
vision tower : CUDNN_SDPA     27/27   calls  (0 eager fallback)
```

## 42. Complete edit time decomposition

`TOTAL_PROCESS = 208.145 s`.

```
MODEL_LOAD              21.741 s   (CPU; must not be counted as inference)
FILE_READ               14.524 s   (CPU, 759 tensors, 35.2 GB, 2.425 GB/s)
CUDA_ALLOC               1.823 s
REF_PREPROCESS           1.041 s
PROMPT_TOKENIZE          0.166 s
INITIAL_NOISE            0.272 s
INPUT_PREPARE            3.935 s
DENOISE_TOTAL          180.195 s
OUTPUT_RECONSTRUCTION    0.141 s
IMAGE_ENCODE_WRITE       0.332 s
REQUEST_TOTAL          207.795 s
TOTAL_PROCESS          208.145 s
```

One-shot vision (inside `INPUT_PREPARE`):

```
VISION_TOTAL             0.7777 s   = 0.37 % of TOTAL_PROCESS
  VISION_PATCH_EMBED     0.5301 s   (cold; includes one-time GEMM tuning — see 43)
  VISION_PREPROCESS      0.0004 s
  VISION_27_BLOCKS       0.2465 s
  VISION_DEEPSTACK_MERG  0.0358 s
  VISION_FINAL_MERGER    0.0007 s
```

Vision 27-block aggregate (0.2465 s):

```
LayerNorm              0.0014 s
QKV GEMM               0.1176 s
RoPE/split/merge       0.0025 s
SDPA                   0.0073 s
O projection           0.0105 s
FC1                    0.0494 s
GELU                   0.0020 s
FC2                    0.0191 s
copies/residual        0.0366 s (residual)
```

Repeated denoise (28 steps):

```
DENOISE_TOTAL          180.195 s
MEAN_DENOISE_STEP        6.434 s
TRANSFORMER_PER_STEP     6.434 s
BLOCKS_TOTAL           179.247 s
SCHEDULER total          0.0140 s  (0.0005 s/step)
FINAL_NORM_HEAD          0.4985 s
EMBEDDING                0.3897 s  (E_xembed 0.1887 s)
```

## 43. Patch-embed measurement caveat

`VISION_PATCH_EMBED` = 0.5301 s in the canonical run vs 0.0305 s in the
pre-Phase-4 baseline. The new patch-embed shape `520x1152x1536` is unseen by the
cuBLASLt tuner, so the first `hd_linear` on it pays a one-time tuning cost that
lands **inside** the timed region.

```
cold            = 0.5301 s
steady state    = 0.0305 s  (0.125 ms/call x 1 call region-equivalent)
measurement includes possible one-time GEMM tuning
```

Not investigated further: patch embed is 0.25 % of total edit runtime.

## 44. Vision hot-path defects fixed

1. **cuDNN vision SDPA was silently falling back to eager.** `hd_sdpa_create`
   always attaches an additive bias tensor, but `hd_sdpa_execute` binds
   `plan->bias` only `if (mask)`. The vision caller passed `mask = NULL` →
   unbound tensor → cuDNN error → eager fallback on all 27 blocks. Fix: pass the
   (zeroed) mask. `VIS_SDPA_CUDNN` 0 → 27.
2. Production `CUDA_STAGE_CHECK` / `cudaDeviceSynchronize` removed from
   `hd_vision_forward`.
3. No `cudaMalloc`/`cudaFree` inside `hd_vision_forward`.
4. No H2D construction of pos-interp/rotary tables inside the forward — moved to
   persistent device storage built once by `hd_vision_prepare_tables()`
   (`ws->pos_idx/pos_wgt/rot_coords/rot_inv`).
5. Vision SDPA plan cached by actual N.
6. Eager attention mask memset skipped when cuDNN vision SDPA is active.
7. Scalar `hd_vision_patch_kernel` replaced by the production GEMM backend
   (`hd_linear`, `X[n,1536] x W[1152,1536]^T + bias`, bf16 in/out, fp32 acc).
   Old kernel preserved behind `HD_VISION_PATCH_REF=1` for reference/A-B.

Vision before → after (GPU seconds):

```
                     before    after
VISION_TOTAL         0.7991    0.7777
VISION_27_BLOCKS     0.7676    0.2465   (-67.9 %)
V_b_qkv_gemm         0.5255    0.1176   (-77.6 %)
V_b_fc1              0.1576    0.0494   (-68.7 %)
V_b_sdpa             0.0079    0.0073
V_b_layernorm        0.0015    0.0014
V_b_o_proj           0.0110    0.0105
```

## 45. x_embedder reference-embedding cache

Confirmed: every denoise step recomputed `x_embedder(target z + static
ref_patches)` over all 8128 rows, although the 4032 reference rows are constant
across steps. Implemented one-time `ref_emb = x_embedder(ref_patches)` plus a
per-step `target_emb = x_embedder(z)` and concatenation.

```
E_xembed    0.7802 -> 0.1887 s   (-75.8 %)
```

Correctness: bit-exact in both backends (probe: `differing=0/33292288` reference,
`differing=0/16777216` production). The cache is hash-neutral — three same-binary
A/B runs (cache ON / `O1_XE_FULL=1` / `O1_XE_NOCACHE=1`) all produced the
identical PNG.

## 46. Output-hash change and its root cause

The final PNG changed (`3a037834…` → `21ca624b…`) after this work. Root cause is
the **patch-embed GEMM replacement**, not the x_embedder cache:

```
HD_VISION_PATCH_REF=1 (scalar kernel) -> b7784b7c…   (= baseline hash)
patch-embed GEMM (production)         -> 21ca624b…   (independent of cache on/off/full)
```

The GEMM patch embed has a tiny per-element error (`cos=1.000000`,
`nrmse=0.000117`) which is chaotically amplified across 28 diffusion steps.
The GEMM path is numerically *better* against the oracle and passes
`make test-vision` 9/9; the output change is expected and documented.

## 47. Makefile header-dependency fix (build correctness)

`make test-sdpa-forward` appeared to regress to 2/4 (`block: scratch too small`).
Root cause was **not** a Phase-4 code defect: the Makefile had no header
dependency tracking, so `tests/unit/test_sdpa_forward.o` and `src/model/block.o`
were stale objects compiled against the pre-`xe_ref` `forward.h`. `block_scratch_bytes`
had shifted offset, so the test wrote it to the old offset and the block read 24
instead of 4 095 360.

Fix: `-MMD -MP` on all compile rules plus `-include` of the generated `.d` files;
`*.d` added to `.gitignore`. Clean rebuild → **4/4 PASS**; touching `forward.h`
now correctly triggers a recompile.

## 48. Correctness ladder (all green)

```
make test-vision        9 passed, 0 failed
make test-sdpa-block    4 assertions passed, 0 failed
make test-sdpa-forward  4 assertions passed, 0 failed
make test-gemm-smoke    builds OK
make test-full-forward  complete_output nrmse=0.15267 (known pre-existing, unchanged)
patch_embed             cos=1.000000 nrmse=0.000117
```

## 49. Edit block profile (GPU, 1008 calls, 177.77 ms/block)

```
region            total s   % block   % denoise
B_sdpa            56.4711    31.51     31.34
B_mlp_gate_up     56.8261    31.71     31.54
B_mlp_down        27.9232    15.58     15.50
B_qkv_proj        14.7708     8.24      8.20
B_o_proj           9.3578     5.22      5.19
B_swiglu           3.0987     1.73      1.72
B_qk_norm          3.0153     1.68      1.67
B_head_split       1.9554     1.09      1.09
B_rope             1.9103     1.07      1.06
B_mlp_resid        0.9792     0.55      0.54
B_attn_resid       0.9786     0.55      0.54
B_post_norm        0.6109     0.34      0.34
B_input_norm       0.6098     0.34      0.34
B_final_copy       0.5752     0.32      0.32
B_mrope            0.0864     0.05      0.05
```

GEMM share: `B_mlp_gate_up + B_mlp_down + B_qkv_proj + B_o_proj` = 108.878 s =
**60.7 % of denoise**. SDPA = 31.3 %.

## 50. Edit decoder SDPA is NOT on the C4 two-pass path

`hd_engine_generate_ref` (generate.c:218) builds its attention plan with the
legacy single masked graph:

```c
hd_sdpa_create(&plan, 1, NH, NKV, S, S, HD, attn_scale);   /* generate.c:671 */
```

C4's two-pass `hd_sdpa_create_prod(..., ar_len, ...)` is used only by
`hd_engine_generate_t2i` (generate.c:1125). The edit mask is more complex than
the t2i mask — token rows are types 1 (target), 2 (reference) and 3 (tms), each
fully unmasked, interleaved with text rows that are causal — so the simple
`ar_len = text_len - 1` split does not apply as-is. The edit path therefore
pays the full masked-graph cost: 56.47 s of SDPA, the single largest edit
region.

## 51. Decision

```
VISION_TOTAL = 0.7777 s = 0.37 % of TOTAL_PROCESS  (threshold 10 %)
=> STOP vision optimization.  No LayerNorm/GELU fusion.
decoder attention = CUDNN_SDPA, no eager fallback  => no emergency.
```

**Largest remaining edit bottleneck: the masked single-graph decoder SDPA
(56.47 s, 31.3 % of denoise)** — the edit path never received C4. GEMMs remain
the larger aggregate at 60.7 % but are already at the device ceiling (Part 2)
and are closed. Next target candidate: extend the two-pass SDPA split to the
edit mask structure.
