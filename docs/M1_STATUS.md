# M1 Execution Status

<!-- Maintain continuously. Must always show the fields below. -->

- **Current sub-step:** M1.4 (whole-model forward, first Python forward)
- **Last green gate:** M1.4 STARTED — entry state verified (check-env/check-locks PASS, oracle `3237a63` clean, Dev `b6acc2f` downloaded, M1.3 fixture PASS), capture under construction
- **Engine HEAD:** `a12e254`
- **Oracle SHA:** `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`
- **Dev model revision:** `b6acc2fe452b3120430620dc4354fa442ee081ea`
- **Base model revision / download status:** `0b0901d99f200389e138c61946af1185f5f49a13` — `not_downloaded`
- **FAST_VALIDATION_RES:** `64×64` (2×2 patches, 4 image tokens, seq 23)
- **Oracle runs consumed (V2+):** 0
- **Native runs consumed (V2+):** 0
- **Known failures:** none
- **Exact next action:** M1.4. One single whole-model Python forward capturing all diagnostic checkpoints in the same run (embedding, block 0, block middle, block last, final norm, final head, complete model output) — not three separate captures. Freeze goldens from that run, then wire the native device-resident path (block bindings + persistent workspace already in place) behind a host orchestrator and validate parity by block. Localize first divergence with the captured checkpoints at the cheapest level; do not rerun Python unnecessarily.

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
| M1.4 whole forward | PENDING | — |
| M1.5 scheduler / 1–3 step | PENDING | — |
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
