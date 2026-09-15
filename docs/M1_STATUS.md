# M1 Execution Status

<!-- Maintain continuously. Must always show the fields below. -->

- **Current sub-step:** M1.1 (weight ingestion and deterministic buffers)
- **Last green gate:** M1.0 (native profile/config/manifest loader)
- **Engine HEAD:** `279316e` (M1.0 to be committed next)
- **Oracle SHA:** `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`
- **Dev model revision:** `b6acc2fe452b3120430620dc4354fa442ee081ea`
- **Base model revision / download status:** `0b0901d99f200389e138c61946af1185f5f49a13` — `not_downloaded`
- **FAST_VALIDATION_RES:** `64×64` (2×2 patches, 4 image tokens, seq 23)
- **Oracle runs consumed (V2+):** 0
- **Native runs consumed (V2+):** 0
- **Known failures:** none
- **Exact next action:** commit guardrails, then M1.0 native loader.

## Gate status

| Gate | Status | Evidence |
|------|--------|----------|
| M0 (baseline) | PASS | `891ed61`; locks/manifests consistent |
| M1 preflight (guardrails) | PASS | `tools/m1_guard.py` (`check-env`/`check-locks` green) |
| M1.0 native loader | PASS | `make test`; dev 759/759, param_count exact, base parses, unknown/missing fail closed |
| M1.0 native loader | PENDING | — |
| M1.1 weight ingestion | PENDING | — |
| M1.2 CUDA primitives | PENDING | — |
| M1.3 decoder block | PENDING | — |
| M1.4 whole forward | PENDING | — |
| M1.5 scheduler / 1–3 step | PENDING | — |
| M1.6 tokenizer | PENDING | — |
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

## Commands failed

- (none)

## Unresolved issues

- (none blocking preflight)

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
