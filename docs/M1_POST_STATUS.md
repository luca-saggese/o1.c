# M1-post Status

## Current sub-step
M1-post COMPLETE — all M1-post.1→.12 workstreams closed. Final 28-step Dev 1024² generation executed and validated.

## Last completed gate
Final 28-step Dev 1024² generation (seed 123456, teapot prompt): `artifacts/m1/final_sanity/dev_native_seed123456.{png,json,log}` — 28/28 steps, no NaN/Inf, final z range [-1.1797, 1.2578], PNG 1024×1024 RGB written, 5/5 assertions PASS. Engine commit at generation time: 13be974.

## Frozen identities
- Oracle SHA: `3237a638a5c2c7be106b0175958f4c0db8c2dfbf` (config/oracle.lock)
- Dev revision: `b6acc2fe452b3120430620dc4354fa442ee081ea` (config/models.lock)
- Base revision: `0b0901d99f200389e138c61946af1185f5f49a13` (config/models.lock)
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
- `./build/sanity_gen --steps 28` — final 28-step Dev 1024² generation: 28/28 steps, PNG + JSON + log under `artifacts/m1/final_sanity/`
- `make test-seq-ref` — ref-mode sequence builder parity vs oracle ground truth (K=1, K=2)

## Commands failed
- `./build/sanity_gen --steps 28` (first attempt): illegal memory access in `hd_attention_eager` at S=1077 — fixed (missing `×2` in `hm` cudaMalloc)
- `./build/sanity_gen --steps 1` (first attempt): segfault in `hd_bf16_buf_to_f32` on device pointer — fixed (stage bf16 to host first)

## Unresolved issues / blockers
1. Skeleton conditioning: **SUPPORTED_BY_UPSTREAM / IMPLEMENTATION_REQUIRED** — official upstream main declares IP-pipeline skeleton support (Multi-Reference Subject-Driven Personalization with Skeleton: face/background/openpose/part refs via `--ref_images`). No OpenPose parser in frozen oracle; semantics carried by generic multi-ref path. Native engine routes skeleton refs through `hd_seq_build` multi-ref path. Whether refs are differentiated by order/metadata/filename/content only: still under audit.
2. Storyboard: **ADVERTISED_CAPABILITY / SEMANTICS_NOT_YET_ESTABLISHED** — official README advertises it; no dedicated API/pipeline in frozen oracle. Audit continues in README, technical report, assets/examples, prompt-agent/web workflow.
3. Per-step noise RNG: CUDA Philox in oracle; native uses CPU MT19937 stand-in (documented).
4. Native tokenizer (297-token subset) cannot encode the teapot prompt → 1024² sequence fixture from oracle pure functions (fixture `/tmp/sanity_seq` complete, all 15 checks PASS).

## Next action
None — M1-post COMPLETE. Hand off to M2 candidates (see docs/M2_CANDIDATES.md).
