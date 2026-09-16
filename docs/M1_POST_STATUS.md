# M1-post Status

## Current sub-step
M1-post.0 — upstream feature audit + capability matrix (in progress → commit)

## Last completed gate
M1 (numerical correctness) — closed. M1-post.0 audit complete.

## Frozen identities
- Oracle SHA: `3237a638a5c2c7be106b0175958f4c0db8c2dfbf` (config/oracle.lock)
- Dev revision: frozen in config/models.lock
- Base revision: frozen in config/models.lock
- Engine commit: see `git log -1 --oneline`

## Audit outputs (artifacts/m1post/audit/, git-ignored)
- A_UPSTREAM_FEATURE_AUDIT.md — core modes
- B_SCHEDULER_PROFILE_AUDIT.md — scheduler/profiles
- C_IMAGE_IO_AUDIT.md — image I/O
- D_CLI_RUNTIME_AUDIT.md — CLI/runtime
- E_CONDITIONING_PARSERS_AUDIT.md — parsers
- F_PROMPT_STORYBOARD_AUDIT.md — prompt/storyboard/progress
- G_TEST_TOOLING_AUDIT.md — tooling

## Frozen docs (committed)
- docs/M1_POST_CAPABILITY_MATRIX.md
- docs/M1_POST_MODE_CONTRACTS.md
- docs/M1_POST_SCHEDULER_MATRIX.md

## Commands executed
- 7 parallel audit subagents (A–G) over frozen oracle source
- Direct verification of DEFAULT_TIMESTEPS, find_closest_resolution, keep_original_aspect

## Commands failed
- (none)

## Unresolved issues / blockers
1. Skeleton conditioning: no OpenPose parser in audited source → UNSUPPORTED unless upstream defines it.
2. Storyboard: no multi-panel decomposition in audited source → UNSUPPORTED.
3. Per-step noise RNG: CUDA Philox in oracle; native uses CPU MT19937 stand-in (documented).

## Next action
Commit M1-post.0 gate (`docs(m1-post): freeze complete hidream feature surface`), then start M1-post.1 (production native T2I + seed + output path) — in progress via sanity runner.