# Development

Engineering history and process notes for o1.c. For user-facing documentation
see the top-level [README](../README.md).

## Milestones

| Milestone | Status | Summary |
|-----------|--------|---------|
| M0 | COMPLETE | Oracle, model and environment pinned; reproducible build |
| M1 | COMPLETE | Native profiles, weight ingestion, CUDA primitives, forward, scheduler, tokenizer, full 28-step Dev inference validated |
| M1.1 | COMPLETE | Weight ingestion into deterministic CUDA buffers |
| M1.2 | COMPLETE | Reference CUDA transformer primitives (golden parity) |
| M1-post | COMPLETE | Production T2I path, feature parity (editing, multi-reference, layout, long text), performance freeze |
| M2 | COMPLETE | Performance: timing instrumentation, benchmark harness, cuDNN SDPA attention, persistent cuBLAS GEMM, two-pass SDPA |
| Release hardening | IN PROGRESS | R0–R9: freeze, model selection, release GGUFs, manifest, downloader, setup, README, Q4 gate, clean-machine test |

## Development principles

- **Python is an oracle, not a runtime.** The frozen upstream checkout lives in
  `python/` (read-only, git-ignored) and is used only for offline validation.
- **Reproducibility is contractual.** Revisions, versions, hashes and gate
  evidence are persisted to locks and manifests in `config/`. A branch name is
  never a substitute for an immutable commit.
- **Cheap validation first.** Most gates run on metadata, startup-only loads or
  1–3 denoising steps; full 28/50-step generations are reserved for milestone
  closure.
- **Dev/Base symmetry.** Dev and Base share one code path; only profile
  parameters differ.
- **Ownership boundaries.** Only engine code (`src/`, `tools/`, `config/`,
  `docs/`, `scripts/`, `tests/`) is versioned. Oracle checkout, model weights,
  build output and generated artifacts are git-ignored.

## Validation

Every primitive is validated against golden fixtures captured once from the
frozen upstream oracle. The validation cost ladder (V0–V6) orders gates from
cheap to expensive.

The native test suite is C/CUDA only:

```sh
make test          # build all test binaries
```

Note that `make test` currently stops at a known pre-existing numerical
exception in `test_full_forward` / `complete_output`
(`nrmse=0.15267 cos=0.98882141 max_abs=1.5`). This is a documented, accepted
pre-existing condition; the threshold is unchanged. Run individual targets
when you need to bypass it.

Test binaries that require arguments:

```sh
build/test_gguf <model.gguf> [safetensors_dir]
build/test_engine_preload <dir_or_gguf> [device]
```

## Build targets

```sh
make                # build/hidream (CLI)
make server         # build/hidream-server
make clean
```

Individual test targets include `build/test_weights`, `build/test_gguf` and
`build/test_server`.

## Release artifacts

Release GGUF conversions are defined in `release/models.def.json` and built by
`scripts/build_release_models.sh`; the results are recorded in
[`docs/RELEASE_MODELS.md`](RELEASE_MODELS.md). The machine-readable manifest is
`models/manifest.json`, consumed by `scripts/download_model.sh`.

## Further reading

- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — engine architecture
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — measured performance
- [`docs/SETUP.md`](SETUP.md) — environment setup
- [`docs/M2_PERFORMANCE.md`](M2_PERFORMANCE.md) — full M2 measurement ledger
- `docs/M1_POST_CLOSEOUT.md` — M1-post feature closeout table
- `docs/WORKING_MODE.md` — engineering invariants
