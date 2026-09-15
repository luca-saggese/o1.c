# M1 Execution Status

<!-- Maintain continuously. Must always show the fields below. -->

- **Current sub-step:** M1.7 — M1 closure: full Dev inference (V5) + Base compatibility
- **Last green gate:** M1.5 scheduler — `test_m1_5_scheduler` 18/18 assertions PASS. Part A isolated scheduler arithmetic bit-exact vs golden z_next (3 steps); Part B V4 1-step parity gate PASS (model_output NRMSE 0.0032 cos 0.99999, z_next NRMSE 0.00043 cos 0.9999999); Part C V4 3-step chain PASS (step1 mo 0.00107/0.9999994, zn 0.00028/0.99999996; step2 mo 0.00158/0.9999988, zn 0.00055/0.99999985). Commit `19c41bd` pushed.
- **Engine HEAD:** `19c41bd` (`feat(m1): match hidream scheduler and deterministic denoising state`)
- **Oracle SHA:** `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`
- **Dev model revision:** `b6acc2fe452b3120430620dc4354fa442ee081ea`
- **Base model revision / download status:** `0b0901d99f200389e138c61946af1185f5f49a13` — `not_downloaded`
- **FAST_VALIDATION_RES:** `64×64` (2×2 patches, 4 image tokens, seq 23)
- **Oracle runs consumed (V2+):** 0
- **Native runs consumed (V2+):** 0
- **Known failures:** none (M1.4 residual fully explained as bf16 drift amplification, contract §6)
- **M1.5 critical finding (timestep domain):** the M1.4 golden's `timestep: 999` field is **scheduler time**, not model input. Frozen oracle converts `step_t=999 → sigma=999/1000=0.999 → model_timestep=1-sigma≈0.001 → embedding input=model_timestep×1000≈1.0`. The M1.4 native passed 999.0 directly to `hd_forward`, giving embedder input 999000 — wrong by ~6 orders. M1.5 must pass `model_timestep≈0.001` to `hd_forward` (which multiplies ×1000 internally). Manifest fields renamed: `scheduler_timestep`/`sigma`/`model_timestep`/`timestep_embedder_input`.
- **Exact next action:** M1.7 Dev closure: define quantitative tolerances for the full 28-step generation (V5), run one native 28-step generation with canonical prompt/seed, compare against frozen oracle trajectory; then Base compatibility (V3/V4 one-forward) — Base revision `0b0901d9…` is `not_downloaded`, STOP + notify if download required.

## Gate status

| Gate | Status | Evidence |
|------|--------|----------|
| M0 (baseline) | PASS | `891ed61`; locks/manifests consistent |
| M1 preflight (guardrails) | PASS | `tools/m1_guard.py` (`check-env`/`check-locks` green) |
| M1.0 native loader | PASS | `make test`; dev 759/759, param_count exact, base parses, unknown/missing fail closed |
| M1.1 weight ingestion | PASS | `make test`; inventory 759/759 exact, 17 probe fingerprints == oracle, 759 CUDA allocs / 35,219,551,168 B, cleanup verified |
| M1.2 CUDA primitives | PASS | `3468ab3`; `make test` 33+18 fixture green, bf16/class B/C/D |
| M1.3 decoder block | PASS | `0519b21`; block_out NRMSE 0.00527 cos 0.999988, 5 internals class D |
| M1.3a arch freeze | PASS | `9bd7c12`; 3 contract docs + shape_inventory/buffer_plan JSON; 0 new Python forwards |
| M1.3b device resident | PASS | `36e4a90`; block bindings resolved once, no strcmp scan / no H2D / no sync in block; block_out NRMSE 0.00527 cos 0.999988 == golden, golden NOT regenerated |
| M1.4 whole forward | PASS | `test_full_forward` 14/14 (block_last 0.0246/0.99996, final_norm 0.0871/0.9963, complete 0.1409/0.9909 within contract §6 envelope); tail machinery self-consistency nrmse=0 exact; `tools/m1_4_drift_predict.py` ratio 1.000/1.001 |
| M1.5 scheduler / 1–3 step | PASS | `19c41bd`; `test_m1_5_scheduler` 18/18 (Part A bit-exact, Part B 1-step, Part C 3-step chain); sched.cu + scheduler.{h,c} + test committed, pushed |
| M1.6 tokenizer | PASS | committed; 17 frozen IDs exact |
| M1.7 Dev closure | PENDING | — |
| M1.7 Base compatibility | PENDING | — |

## Commands executed

- `git status --short`, `git log --oneline` — verified clean baseline.
- `git -C python status --short`, `rev-parse HEAD` — oracle clean at `3237a638…`.
- Confirmed scheduler discrepancy: Dev T2I non-editing uses `flash` scheduler,
  not `flow_match`; corrected `tools/freeze_startup.py::_scheduler_freeze` and
  regenerated the manifest scheduler block (commit `c937fa6`). Structural
  timesteps/sigmas unchanged (same DEFAULT_TIMESTEPS, shift 1.0); only
  `step()` semantics differ.
- Computed `flash` vs `flow_match` structural config offline (no model load):
  identical 28 timesteps / 29 sigmas.
- `make clean && make` — built native loader (`build/hidream`).
- `./build/hidream --model dev|base|foo` — verified dev/base parse, unknown
  fails closed.
- `make test` — 24/24 unit assertions pass (dtype, numel, manifest load,
  identical/mismatch compare, unknown profile, path-traversal rejection).
- `./build/hidream --model dev --inventory --probe` — V0 host-only run:
  expected=759 found=759 missing=0 unexpected=0; shape/dtype/numel mismatches 0;
  total_bytes=35,219,551,168; largest=`lm_head.weight` (622,329,856).
- Oracle cross-check of all 17 probe fingerprints via `.venv/bin/python` +
  `safetensors.safe_open`: every SHA-256 matches the native reader byte-for-byte.
- `./build/hidream --model dev --to-device` — 759 tracked `cudaMalloc` buffers,
  device_bytes_allocated=35,219,551,168 == host_bytes_loaded, all freed (13.6 s).
- `make test` — `tests/unit/test_weights.c` adds 77 assertions (index parse,
  sorted table, inventory, oracle fingerprint table, device placement,
  cleanup + repeat placement, Base profile intact). Total 101 assertions pass.
- `.venv/bin/python tools/m1_guard.py check-locks` — `ok: true`, no problems.
- `make test-full-forward` — M1.4 whole-forward gate: 14/14 assertions PASS.
  embedding/target_embedding/timestep_conditioning/block_0/block_mid class D;
  block_last 0.0246/0.99996, final_norm_input 0.0246, final_norm 0.0871/0.9963,
  final_head_input 0.0871, complete_output 0.1409/0.9909 — all within the
  contract §6 drift-amplification envelope (3e-2 / 0.1 / 0.18).
- `./build/test_block_tail` — isolated-tail harness: machinery self-consistency
  (seed=native block_mid) nrmse=0 cos=1 exact; golden-mid seed → native
  layers 19–35 vs golden block_last nrmse=0.0098 cos=0.99997 (class D PASS);
  norm+head self-consistency on golden block_last: final_norm 0.0029,
  complete 0.0041 (bf16 storage floor).
- `.venv/bin/python tools/m1_4_drift_predict.py` — oracle RMSNorm+head formula
  (fp64) on native block_last predicts final_norm 0.087121 vs observed
  0.087127 (ratio 1.000) and complete_output 0.14102 vs 0.14094 (ratio 1.001).
  Proves norm/head correct; residual is bf16 input-drift amplification.

## Commands failed

- `make` initially failed compiling `src/model/weights.c` (`cudaDeviceProp` needs
  the `struct` keyword when only `cuda_runtime_api.h` is visible in C mode).
  Fixed with `struct cudaDeviceProp`; also removed an unused `dtype_size()` and
  silenced the unused `model_dir` parameter.
- First `--inventory --probe` run reported 755 missing / 755 unexpected: the
  versioned manifest preserves discovery order, not name order, while the shard
  index is name-sorted. `hd_weights_inventory` now sorts a pointer array over
  the manifest before the merge. The manifest file itself is left untouched so
  its M0 fingerprint stays valid.

## Unresolved issues

- (none blocking)

## M1.1 gate evidence

| Gate condition | Result |
|----------------|--------|
| V0 PASS | `make test` 101 assertions, `--inventory` exit 0 |
| 0 transformer forwards | no CUDA kernel launched; only `cudaMalloc`/`cudaMemcpy`/`cudaDeviceSynchronize` |
| all expected Dev weights resolved | 759/759, missing=0 |
| no unexpected tensor silently ignored | unexpected=0 |
| raw values/fingerprints match source | 17/17 SHA-256 identical to `safetensors` oracle |
| CUDA allocations succeed | 759 allocations, 35,219,551,168 B |
| cleanup frees all owned resources | `hd_weight_store_free` then repeat placement reproduces identical accounting |
| Base profile path remains supported | `hd_profile_load("base")` ok, path `models/base`, immutable revision |
| host-only parser tests | `tests/unit/test_weights.c`

## Run budget (target)

| Stage | Python oracle work |
|-------|--------------------|
| M1.0 | 0 forwards |
| M1.1 | 0 forwards |
| M1.2 | primitive/module calls only |
| M1.3 | one decoder-block invocation |
| M1.4 | 1 whole forward |
| M1.5 | 1 three-step run (=3 forwards) |
| M1.6 | 0 forwards |
| M1.7 Dev | 1 full 28-step generation |
| M1.7 Base | 1 whole forward (no V6) |
