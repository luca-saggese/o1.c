# M1-post Status

## Current sub-step
M1-post.1 — production native T2I path (request ABI + sequence builder done; 1024² fixture + native runner complete; final 28-step Dev generation pending as last step)

## Last completed gate
M1-post.1 runner: native 1024² generation runner (`tests/unit/sanity_gen.c`) builds and runs. Fixed critical `hd_attention_eager` buffer under-allocation in `src/cuda/attn.cu` (cudaMalloc for `hm` was `heads*seq*dim` bytes, missing `×2` for bf16 — caused illegal memory access at S=1077). 1-step decode+PNG path validated: PNG 1024×1024 RGB decodes, no NaN/Inf, range sane. Base local-layer-18 decisive test PASS (E_local_18=9.25e-05 → block_mid NRMSE=0.010551 classified as accumulated upstream drift, not a layer-18 defect).

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
- `make test-seq` — sequence parity gate vs M1.4 fixture: pos_f32/mask/vinput_mask byte-identical (4/4 PASS)
- `make test-sanity` — native 1024² runner builds clean
- `./build/sanity_gen --steps 1` — 1-step decode+PNG validation: PNG 1024×1024 RGB, no NaN/Inf, range sane
- `make test-m17-base-local` — E_local_18=9.25e-05 PASS (layer 18 internally correct; 0.010551 = accumulated drift)

## Commands failed
- `./build/sanity_gen --steps 28` (first attempt): illegal memory access in `hd_attention_eager` at S=1077 — fixed (missing `×2` in `hm` cudaMalloc)
- `./build/sanity_gen --steps 1` (first attempt): segfault in `hd_bf16_buf_to_f32` on device pointer — fixed (stage bf16 to host first)

## Unresolved issues / blockers
1. Skeleton conditioning: no OpenPose parser in audited source → UNSUPPORTED unless upstream defines it.
2. Storyboard: no multi-panel decomposition in audited source → UNSUPPORTED.
3. Per-step noise RNG: CUDA Philox in oracle; native uses CPU MT19937 stand-in (documented).
4. Native tokenizer (297-token subset) cannot encode the teapot prompt → 1024² sequence fixture from oracle pure functions (fixture `/tmp/sanity_seq` complete, all 15 checks PASS).

## Next action
Run the final 28-step Dev 1024² generation (seed 123456, teapot prompt) → PNG + JSON + log under `artifacts/m1/final_sanity/`, then mark M1-post COMPLETE.
## M1-post integration phase (current sub-step)
- Sub-step: M1-post.1→.12 implementation (parallel workstreams)
- Last completed gate: attention fix + 1-step decode/PNG validation (b17cbcc)
- Current engine commit: b17cbcc
- Plan: parallel subagents for tokenizer-full / image-pipeline / scheduler-matrix; integrator (main agent) owns unified sequence builder ref modes + CLI generation path. GPU/oracle runs serialized.

## M1-post integration phase (current sub-step)
- Sub-step: M1-post.1→.12 implementation (parallel workstreams)
- Last completed gate: attention fix + 1-step decode/PNG validation (b17cbcc)
- Current engine commit: b17cbcc
- Plan: parallel subagents for tokenizer-full / image-pipeline / scheduler-matrix; integrator (main agent) owns unified sequence builder ref modes + CLI generation path. GPU/oracle runs serialized.
