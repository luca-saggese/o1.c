# HiDream GB10 inference engine — M0–M2 plan

## Scope and non-negotiable implementation constraints

The target is a HiDream-O1-Image inference engine with a C-facing runtime and CUDA backend, specialized for NVIDIA GB10. The project starts from the official upstream implementation only as a frozen Python oracle; production inference must not depend on Python.

Three constraints are architectural, not temporary conveniences:

1. **Minimize expensive model runs.** Most development gates must be validated with metadata checks, startup-only loading, frozen intermediate tensors, one forward, or 1–3 denoising steps. A full 28-step Dev generation is reserved for milestone closure or a regression that cannot be localized more cheaply. A 50-step Base/Full generation is never a routine per-commit test.
2. **Official Python oracle lives in `/python` relative to the project root and is git-ignored.** It is pinned by upstream commit SHA and treated as read-only. Local modifications invalidate the oracle until re-frozen.
3. **Dev first, Dev/Base always.** HiDream-O1-Image-Dev-2604 is the primary development checkpoint. The runtime API, tensor loader, model config and scheduler abstraction must keep both `dev` and `base`/`full` profiles valid from the first implementation commit. No Dev-only hardcoded tensor topology is allowed where Base differs by configuration.

The upstream `dev` branch is the chosen oracle branch because its current `inference.py` still exposes both `full` and `dev` model types, while Dev-2604 is published from that branch.

## Validation cost ladder

Every test or gate is assigned the cheapest level that can prove the required property.

| Level | Model work | Intended use |
|---|---|---|
| V0 | none | parsing, configs, file layout, tensor metadata, scheduler tables |
| V1 | startup/load only | processor/model construction, state dict manifest, memory footprint |
| V2 | one selected layer/block | kernel parity, tensor layout, normalization/RoPE/GEMM checks |
| V3 | one whole model forward | end-to-end transformer numerical parity without denoising loop |
| V4 | 1–3 denoising steps | scheduler/state evolution and cross-step correctness |
| V5 | full Dev generation (28 steps) | milestone/regression closure only |
| V6 | full Base generation (50 steps) | release/major compatibility gate only |

Rules:

- A failing V2/V3/V4 gate must be debugged at the same or lower level; do not rerun V5 repeatedly.
- Golden tensors are captured once from the frozen oracle and versioned only when the oracle revision or deliberate numerical contract changes.
- Golden files should contain selected tensors/statistics, not every activation of every layer.
- All comparisons use fixed seed, fixed prompt and a fixed validated resolution. The first oracle checkout determines the lowest resolution accepted without changing upstream semantics; that resolution becomes the fast validation resolution.
- Full-resolution 2048×2048 runs belong to performance/quality characterization, not routine correctness validation.

---

# M0 — Environment, model acquisition, frozen Python startup

## Goal

Create a reproducible GB10 development environment and freeze a trustworthy Python oracle **without running image generation**.

## M0.0 — Repository scaffold and run-budget policy

**Work**

- initialize project repository;
- add `/python/`, model weights, HF cache, build outputs and generated artifacts to `.gitignore`;
- add the Dev/Base profile configuration template;
- add environment/oracle/model bootstrap scripts;
- document the validation cost ladder and milestone gates.

**Validation:** V0.

**Gate M0.0**

- `/python/` is ignored by Git;
- `models/` is ignored by Git;
- config exposes independent `dev` and `base` model repositories/paths;
- bootstrap scripts fail closed when revisions/config are missing rather than silently selecting moving targets.

**Commit**

`chore(m0): scaffold repo and define low-run validation policy`

## M0.1 — Pin and checkout official Python oracle

**Work**

- checkout `HiDream-ai/HiDream-O1-Image` branch `dev` into `/python`;
- resolve and record exact upstream commit SHA;
- never patch the oracle in place;
- create a local Python environment from the pinned upstream requirements;
- record package versions after a successful import.

**Validation:** V0.

**Gate M0.1**

- `python/.git` exists locally but `git status` of the engine repository does not see `/python`;
- oracle working tree is clean;
- branch/revision are recorded in the freeze manifest/config;
- importing upstream model/pipeline modules succeeds.

**Commit**

`chore(m0): pin official hidream python oracle`

## M0.2 — Acquire and pin model weights

**Primary checkpoint:** `HiDream-ai/HiDream-O1-Image-Dev-2604`.

**Base checkpoint:** `HiDream-ai/HiDream-O1-Image`; supported in configuration from day one, but its large weight download may be deferred until the first Base compatibility gate if storage/network time matters.

**Work**

- resolve immutable Hugging Face revision(s);
- download Dev weights into `models/dev`;
- optionally download Base weights into `models/base`;
- store file names/sizes and hashes of small metadata/configuration files;
- do not hash every multi-GB shard on each development run.

**Validation:** V0.

**Gate M0.2**

- Dev model directory is complete and loadable with `local_files_only=True`;
- revision is immutable and recorded;
- no model file is tracked by the engine Git repository;
- Base profile resolves to a valid repository/path even if its weights are intentionally not yet downloaded.

**Commit**

`chore(m0): pin dev checkpoint and model acquisition workflow`

## M0.3 — Freeze startup oracle

**Work**

Perform **startup only**, not generation:

- load processor/tokenizer;
- load model on CUDA exactly as the frozen oracle defines it initially;
- dump environment versions and GPU compute capability;
- dump state-dict tensor names, shapes, dtypes and parameter count;
- freeze token IDs for a short canonical prompt;
- freeze scheduler/config metadata needed later by the C implementation;
- record peak startup allocation if practical.

No transformer forward is permitted in this step.

**Validation:** V1.

**Gate M0.3 / M0 complete**

- startup succeeds from local files with network disabled;
- manifest captures oracle SHA, model revision, Python/Torch/Transformers/CUDA versions and GPU identity;
- state-dict manifest is stable across two startup-only invocations;
- canonical prompt tokenization is stable;
- Dev and Base are represented as explicit profiles in the frozen configuration;
- **zero complete image-generation runs were needed to close M0.**

**Commit**

`test(m0): freeze reproducible python startup oracle`

---

# M1 — Complete non-optimized C/CUDA inference, validated against Python

## Goal

Produce a correct end-to-end T2I implementation whose control path is native C/CUDA. Correctness has priority over speed. Python may be used offline to export/freeze reference data or convert weights, but must not participate in production inference.

## M1.0 — Native model/profile/config loader

Implement shared `dev`/`base` profile representation, tensor metadata, dimensions, scheduler selection and CLI/API configuration. No model math yet.

**Validation:** V0 against the frozen Python manifest.

**Commit:** `feat(m1): add native hidream model profiles and tensor manifest loader`

## M1.1 — Weight ingestion and deterministic buffers

Implement safetensors ingestion directly or a deliberately simple offline packed-format converter plus native loader. Preallocate deterministic CUDA buffers. Preserve tensor names/layout mapping to the oracle.

**Validation:** V0; compare tensor shapes, element counts and selected raw values/checksums.

**Commit:** `feat(m1): load hidream weights into deterministic cuda buffers`

## M1.2 — Primitive numerical parity

Implement the minimum unoptimized CUDA path for RMSNorm/QK norm, MRoPE, GEMM projections, SwiGLU, attention/GQA, embeddings, patchify/unpatchify and output projection. cuBLAS/cuBLASLt is allowed; optimization-specific fusion is deferred.

**Validation:** V2 on selected layers and synthetic/frozen inputs. Compare max abs/rel error plus cosine similarity; never rely only on image appearance.

**Commit:** `feat(m1): implement reference cuda transformer primitives`

## M1.3 — One decoder block parity

Wire a complete decoder block using frozen real-model inputs/weights.

**Validation:** V2. Gate on block input/output and selected internal tensors.

**Commit:** `test(m1): validate one decoder block against python oracle`

## M1.4 — Whole transformer forward parity

Wire all layers, timestep conditioning and pixel head.

**Validation:** V3: exactly one whole forward at the fast validation resolution. Capture only diagnostic checkpoints needed to localize drift.

**Commit:** `feat(m1): complete native transformer forward path`

## M1.5 — Scheduler and 1–3 step inference parity

Implement Dev and Base scheduler semantics and RNG/noise handling. Establish deterministic initial state interchange so Python and C can start from identical noise tensors.

**Validation:** V4: first 1 step, then 3 steps only after 1-step parity passes.

**Commit:** `feat(m1): match hidream scheduler and deterministic denoising state`

## M1.6 — Native tokenizer/input path

Remove dependence on pre-tokenized golden inputs for normal CLI use. Validate token IDs and special tokens against the frozen oracle.

**Validation:** V0/V1; tokenizer tests require no model forward.

**Commit:** `feat(m1): add native prompt tokenization and special-token handling`

## M1.7 — M1 closure: full Dev inference

Run one controlled full 28-step Dev generation at the validation resolution with the canonical prompt/seed. If numerical differences are expected because of allowed kernel precision, define quantitative tensor tolerances plus perceptual/image metrics before running the gate.

**Validation:** V5, normally one oracle run + one native run. Avoid iterative full-run debugging.

Also run Base compatibility at V3 or V4 when Base weights are available; a 50-step V6 run is not required to close early M1 unless Base-specific logic cannot otherwise be proven.

**Gate M1**

- native CLI accepts both `--model dev` and `--model base`;
- Dev one-forward and 1–3-step numerical gates pass;
- full Dev generation passes the agreed final comparison;
- Base passes at least configuration + weight loading + one-forward compatibility;
- Python is absent from the runtime dependency chain.

**Commit:** `test(m1): validate full dev inference against frozen oracle`

---

# M2 — Python/C performance baseline and GB10 optimization

## Goal

Measure before optimizing, then improve latency without weakening the M1 correctness gates.

## M2.0 — Fair benchmark harness

Benchmark Python oracle and native engine with identical model, dtype, resolution, seed and step count. Split timings into startup/load, first inference, steady-state inference and per-step/forward time. Record GPU power/clocks/memory where practical.

Use shortened step counts for iteration; use full Dev only for milestone measurements.

**Commit:** `bench(m2): add reproducible python-vs-native gb10 baseline`

## M2.1 — Identify dominant kernels

Profile with Nsight Systems/Compute or equivalent. Rank GEMM, attention, normalization, elementwise kernels, launches and memory movement by wall time.

**Commit:** `perf(m2): profile gb10 bottlenecks and freeze baseline report`

## M2.2 — Low-risk execution optimizations

Candidate order, only where profiling supports it:

- persistent/preallocated buffers and zero avoidable host copies;
- cuBLASLt algorithm/layout tuning;
- CUDA Graph capture for fixed shapes;
- fused normalization/RoPE/elementwise operations;
- optimized GQA/flash-attention path;
- prefix/static-conditioning reuse where architecture permits.

Every optimization must pass V2/V3 before any V5 rerun.

**Commit series:**

- `perf(m2): tune gb10 cublaslt projection kernels`
- `perf(m2): capture fixed-shape denoising with cuda graphs`
- `perf(m2): fuse normalization rope and activation kernels`
- `perf(m2): optimize gqa attention for gb10`
- `perf(m2): reuse static conditioning across denoising steps`

## M2.3 — Precision optimization

Only after the BF16/FP32-native path is stable: evaluate BF16 and then FP8 selectively. FP4/NVFP4 is a later experiment, not an M2 prerequisite.

Each precision mode has its own numerical/quality contract and is never compared against a mismatched Python dtype as if it were strict parity.

**Commit:** `perf(m2): add validated reduced-precision execution modes`

## Gate M2

- reproducible before/after benchmark on the target GB10;
- correctness gates from M1 remain green;
- performance improvement is attributed to measured bottlenecks, not anecdotal wall-clock runs;
- Dev remains the primary benchmark and Base retains configuration/forward compatibility;
- final report includes startup, 1-forward, 3-step, full-Dev latency, peak memory and speedup versus Python.

---

# Canonical commit sequence

1. `chore(m0): scaffold repo and define low-run validation policy`
2. `chore(m0): pin official hidream python oracle`
3. `chore(m0): pin dev checkpoint and model acquisition workflow`
4. `test(m0): freeze reproducible python startup oracle`
5. `feat(m1): add native hidream model profiles and tensor manifest loader`
6. `feat(m1): load hidream weights into deterministic cuda buffers`
7. `feat(m1): implement reference cuda transformer primitives`
8. `test(m1): validate one decoder block against python oracle`
9. `feat(m1): complete native transformer forward path`
10. `feat(m1): match hidream scheduler and deterministic denoising state`
11. `feat(m1): add native prompt tokenization and special-token handling`
12. `test(m1): validate full dev inference against frozen oracle`
13. `bench(m2): add reproducible python-vs-native gb10 baseline`
14. `perf(m2): profile gb10 bottlenecks and freeze baseline report`
15. optimization commits driven by the profiler, one independent optimization per commit.

## Merge rule

A commit may move to the next validation cost level only when the cheaper gate is green. In particular, no performance change is accepted solely because a full image “looks right”, and no developer should use repeated 28/50-step generations as a debugging loop.
