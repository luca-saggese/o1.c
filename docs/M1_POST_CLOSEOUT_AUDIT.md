# M1-post Closeout Audit

Engine HEAD: f0f44dc (docs(m1-post): correct status — M1-post.1 done, M1-post.2..12 open)
Oracle SHA: 3237a638a5c2c7be106b0175958f4c0db8c2dfbf (config/oracle.lock)
Full/Base revision: 0b0901d99f200389e138c61946af1185f5f49a13 (config/models.lock)
Dev revision: b6acc2fe452b3120430620dc4354fa442ee081ea (config/models.lock)
Dev-2604 revision: b6acc2fe452b3120430620dc4354fa442ee081ea (config/models.lock, profile "dev")
Audit started: 2026-09-17

Source of truth for this closeout (old tasklist is NOT authoritative):
- M1_POST_FULL_FEATURE_PARITY.md (sections 3, 57–90)
- git history
- already-produced artifacts / goldens
- tests in tests/unit and Makefile targets

documents the parity spec references but which DO NOT EXIST:
- docs/M1_POST_UPSTREAM_FEATURE_AUDIT.md  (MISSING — spec §2 required)
- docs/M1_POST_REPORT.md                  (MISSING — referenced by start instructions)
- artifacts/m1_post/final_sanity/         (MISSING — spec §74; only artifacts/m1/final_sanity exists)

## Status legend

VERIFIED_DONE
PARTIAL
MISSING
NEEDS_EVIDENCE
NOT_APPLICABLE_UPSTREAM
BLOCKED_M2
REGRESSION

---

## Initial inventory from git history (FASE 2)

M1-post commit range: c909300 (freeze) .. HEAD f0f44dc. All 16 commits dated 2026-09-16.

| Area | Commit/evidence | Initial status |
|------|-----------------|----------------|
| upstream feature audit (M1-post.0) | c909300; artifacts/m1post/audit/A–G; docs/M1_POST_CAPABILITY_MATRIX.md | VERIFIED_DONE (spec-named doc missing) |
| request ABI + t2i sequence builder | 140ebdf feat(m1-post): unified request ABI and t2i sequence builder | VERIFIED_DONE |
| dev recipe sigmas | c894178 feat(m1-post): derive dev recipe sigmas from frozen timesteps | VERIFIED_DONE |
| unified output decode | 448b412 feat(m1-post): unified native output decode path | NEEDS_EVIDENCE |
| 1024 native generation | b17cbcc fix: attention scratch + native 1024 runner; artifacts/m1/final_sanity/dev_native_seed123456.png | VERIFIED_DONE |
| production T2I | 4341f14 feat: native image pipeline + generation CLI; 28-step run | NEEDS_EVIDENCE |
| native RNG | 131860d feat(m1): torch-exact CPU RNG with bitwise gate | VERIFIED_DONE |
| 2048 | none | MISSING |
| long text / multilingual | 94501b1 full-vocab tokenizer | NEEDS_EVIDENCE |
| single-ref editing | 2bc625e fix(m1-post): ref-mode sequence builder matches oracle; make test-seq-ref | NEEDS_EVIDENCE |
| keep-original-aspect | none found | NEEDS_EVIDENCE |
| Dev edit flow_match | 9b03c14 scheduler matrix | NEEDS_EVIDENCE |
| Dev edit flash | c894178 | NEEDS_EVIDENCE |
| multi-ref personalization | 2bc625e (K=1,K=2 seq parity) | NEEDS_EVIDENCE |
| many-ref handling | none found | MISSING |
| layout conditioning | 5cd4134 feat(m1-post): native layout conditioning with bit-exact oracle parity | NEEDS_EVIDENCE |
| skeleton conditioning | 4f0a918 reclassified SUPPORTED_BY_UPSTREAM/IMPLEMENTATION_REQUIRED | NEEDS_EVIDENCE |
| Full scheduler/guidance | 9b03c14 scheduler matrix with flash/flow_match/default UniPC parity | NEEDS_EVIDENCE |
| Dev scheduler/guidance | 9b03c14 | NEEDS_EVIDENCE |
| noise controls | 9b03c14 (sched.c) | NEEDS_EVIDENCE |
| storyboard | 4f0a918 reclassified ADVERTISED/UNKNOWN | NEEDS_EVIDENCE |
| prompt agent/refiner | none | MISSING |
| progress callback | src/runtime/request.h has hd_progress_callback; generate.c | NEEDS_EVIDENCE |
| preview callback | none found | MISSING |
| CLI | 4341f14 generation CLI (T2I only) | PARTIAL |
| public C API | include/hidream.h + runtime headers | NEEDS_EVIDENCE |
| invalid-combination handling | src/runtime/request.c hd_request_validate | NEEDS_EVIDENCE |
| offline/Python-free runtime | native C/CUDA, no Python in build | NEEDS_EVIDENCE |
| final sanity artifacts (§74 matrix) | only dev_native_seed123456.png | MISSING |

---

## Definition of Done (transcribed from M1_POST_FULL_FEATURE_PARITY.md, once)

Verification order per FASE 10. Each item uses the fixed structure below.

## ITEM-01 — Capability matrix frozen and complete (spec §3, §78)

Status: NEEDS_EVIDENCE

Evidence:
- commit: c909300 docs(m1-post): freeze complete hidream feature surface
- test:
- artifact: docs/M1_POST_CAPABILITY_MATRIX.md, docs/M1_POST_MODE_CONTRACTS.md, docs/M1_POST_SCHEDULER_MATRIX.md, artifacts/m1post/audit/A–G
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §3, §78

Missing: docs/M1_POST_UPSTREAM_FEATURE_AUDIT.md (spec-named artifact)

Action: confirm capability matrix content; record missing spec-named audit doc.

Last checked:

## ITEM-02 — Production native T2I path (§79, §9)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 140ebdf, b17cbcc, 4341f14
- test:
- artifact: artifacts/m1/final_sanity/dev_native_seed123456.{png,json,log}
- source file: src/runtime/generate.c, src/runtime/sequence.c
- documentation: docs/M1_POST_STATUS.md

Missing:

Action: confirm 28-step full generation artifact + assertions.

Last checked:

## ITEM-03 — Native seed / RNG path (§10, §79)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 131860d feat(m1): add torch-exact CPU RNG with bitwise gate
- test: make test-rng (tests/unit/test_torch_rng.c)
- artifact: artifacts/m1/golden/M1_RNG
- source file: src/runtime/torch_rng.c
- documentation:

Missing:

Action: run make test-rng, confirm bitwise gate.

Last checked:

## ITEM-04 — Native image input decode / output decode / PNG (§48, §54, §79)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 2be7727 feat(m1): vendor iris PNG encoder; 448b412 unified output decode
- test: make test-image (test_image.c), make test-png (test_png_roundtrip.c), make test-decode (test_decode.c)
- artifact:
- source file: src/image/hd_image.c, src/io/png_wrap.c, src/runtime/decode.c
- documentation: artifacts/m1post/audit/C_IMAGE_IO_AUDIT.md

Missing:

Action: run test-decode + test-image + test-png.

Last checked:

## ITEM-05 — 1024² generation works (§79)

Status: NEEDS_EVIDENCE

Evidence:
- commit: b17cbcc
- test: make test-sanity
- artifact: artifacts/m1/final_sanity/dev_native_seed123456.png (1024×1024)
- source file: tests/unit/sanity_gen.c
- documentation: docs/M1_POST_STATUS.md

Missing:

Action: confirm PNG dimensions from artifact JSON.

Last checked:

## ITEM-06 — 2048 model path characterized (§11, §12, §68, §80)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/sequence.c (resolution snapping)
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing: no 2048 run or measured blocker recorded

Action: classify as A PASS / B BLOCKED_M2 / C MISSING; if C, implement resolution path; if B, record S, workspace bytes, failure point.

Last checked:

## ITEM-07 — Resolution snapping / aspect ratios (§11, §12, §80)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md (PREDEFINED_RESOLUTIONS, find_closest_resolution)

Missing: native find_closest_resolution implementation not confirmed

Action: locate native snapping implementation + targeted test.

Last checked:

## ITEM-08 — Long prompts not truncated / Unicode exact at tokenizer (§14, §63, §80)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 94501b1 fix(m1): full-vocab tokenizer with sorted-table binary search
- test: make test-tokenizer
- artifact:
- source file: src/model/tokenizer.c
- documentation:

Missing: long-text/multilingual corpus not frozen

Action: run make test-tokenizer; check full-vocab coverage.

Last checked:

## ITEM-09 — English long-text sanity artifact (§14, §63, §74)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected artifacts/m1_post/final_sanity/long_text_en.png
- source file:
- documentation:

Missing: long_text_en.png not produced; corpus not frozen

Action: freeze corpus then generate artifact.

Last checked:

## ITEM-10 — Chinese long-text sanity artifact (§63, §74)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected artifacts/m1_post/final_sanity/long_text_zh.png
- source file:
- documentation:

Missing: long_text_zh.png not produced

Action: generate from Chinese corpus after ITEM-09 corpus frozen.

Last checked:

## ITEM-11 — Multi-region text / layout semantics (§16, §26–28, §83)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 5cd4134 feat(m1-post): native layout conditioning with bit-exact oracle parity
- test: make test-layout (tests/unit/layout_pipe.c)
- artifact: tests/unit/layout_oracle_dump.txt
- source file: src/image/layout.c
- documentation: docs/M1_POST_MODE_CONTRACTS.md §4

Missing: full layout-conditioned sanity image

Action: run make test-layout; confirm parser + canvas parity.

Last checked:

## ITEM-12 — Single-reference decode / preprocess parity (§18, §17, §81)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 2bc625e fix(m1-post): ref-mode sequence builder matches oracle
- test: make test-seq-ref (tests/unit/test_seq_ref.c)
- artifact:
- source file: src/runtime/sequence.c
- documentation:

Missing:

Action: run make test-seq-ref; confirm K=1 parity.

Last checked:

## ITEM-13 — Edit sequence matches oracle (§81)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 2bc625e
- test: make test-seq-ref
- artifact:
- source file: src/runtime/sequence.c
- documentation:

Missing:

Action: confirm ref-mode sequence parity K=1.

Last checked:

## ITEM-14 — keep-original-aspect (§19, §81)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact: expected artifacts/m1_post/final_sanity/edit_keep_aspect.png
- source file: src/runtime/request.h (keep_original_aspect field)
- documentation: docs/M1_POST_MODE_CONTRACTS.md §2

Missing: implementation/behavior unconfirmed; artifact absent

Action: trace keep_original_aspect through sequence builder; targeted test.

Last checked:

## ITEM-15 — Dev edit flow_match (§20, §81)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 9b03c14 feat(m1-post): scheduler matrix with flash/flow_match/default UniPC parity
- test: tests/unit/scheduler_matrix.c
- artifact: tests/unit/oracle_dump_matrix.txt
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing: end-to-end edit run in flow_match (generate.c currently rejects non-flash)

Action: run scheduler matrix test; note generate.c gate.

Last checked:

## ITEM-16 — Dev edit flash (§81)

Status: NEEDS_EVIDENCE

Evidence:
- commit: c894178, 9b03c14
- test: scheduler_matrix.c
- artifact:
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing: edit-mode end-to-end gated in generate.c

Action: same as ITEM-15.

Last checked:

## ITEM-17 — Full edit passes (§81)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: recipe Full 50 steps default scheduler

Missing: Full profile edit run not found

Action: determine whether generate.c supports Full; targeted test.

Last checked:

## ITEM-18 — Full native edit artifact produced (§76, §81)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected artifacts/m1_post/final_sanity/edit_1ref.png
- source file:
- documentation:

Missing: no edit artifact

Action: produce after edit path verified.

Last checked:

## ITEM-19 — 2-reference parity (§22, §82)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 2bc625e (test-seq-ref K=2)
- test: make test-seq-ref
- artifact:
- source file: src/runtime/sequence.c
- documentation: docs/M1_POST_MODE_CONTRACTS.md §3

Missing:

Action: run make test-seq-ref; confirm K=2 parity.

Last checked:

## ITEM-20 — Multi-reference count scaling / preprocessing by K (§23, §67, §82)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/sequence.c
- documentation: docs/M1_POST_MODE_CONTRACTS.md §3 (K sizing table)

Missing: many-ref stress (>2) not evidenced

Action: targeted many-ref sequence test.

Last checked:

## ITEM-21 — Reference ordering validated (§24, §82)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 2bc625e
- test: make test-seq-ref
- artifact:
- source file: src/runtime/sequence.c
- documentation: docs/M1_POST_MODE_CONTRACTS.md §3

Missing:

Action: confirm ordered concat in parity test.

Last checked:

## ITEM-22 — Subject identity sanity artifact (§76, §82)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected subject_2ref.png / subject_multiref.png
- source file:
- documentation:

Missing: no personalization artifact

Action: produce after multi-ref path verified.

Last checked:

## ITEM-23 — Layout JSON forms parsed + bbox frozen (§27, §28, §65, §83)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 5cd4134
- test: make test-layout
- artifact: tests/unit/layout_oracle_dump.txt
- source file: src/image/layout.c
- documentation: docs/M1_POST_MODE_CONTRACTS.md §4

Missing:

Action: confirm parser coverage of list/wrapper/keyed forms.

Last checked:

## ITEM-24 — Reference-to-box mapping correct (§83)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 5cd4134
- test: make test-layout
- artifact:
- source file: src/image/layout.c
- documentation:

Missing:

Action: confirm mapping test.

Last checked:

## ITEM-25 — Layout one-forward parity (§83, §61)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation:

Missing: no one-forward parity capture for layout mode

Action: build layout fixture through sequence builder; compare to oracle.

Last checked:

## ITEM-26 — Full layout-conditioned sanity (§76, §83)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected subject_layout.png
- source file:
- documentation:

Missing: no layout-conditioned image

Action: produce after ITEM-25.

Last checked:

## ITEM-27 — Skeleton reference-role semantics frozen (§29, §30, §66, §84)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 4f0a918
- test:
- artifact: artifacts/m1post/audit/E_CONDITIONING_PARSERS_AUDIT.md
- source file:
- documentation: docs/M1_POST_MODE_CONTRACTS.md §5, capability matrix blocker 1

Missing: role differentiation (order/metadata/filename/content) unresolved

Action: targeted audit of ref_images preprocessing; decide status.

Last checked:

## ITEM-28 — Skeleton preprocessing parity (§84)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/sequence.c (HD_REF_SKELETON role)
- documentation:

Missing:

Action: verify skeleton ref routed through generic multi-ref preprocessing.

Last checked:

## ITEM-29 — Skeleton sequence parity (§84)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 2bc625e
- test: make test-seq-ref
- artifact:
- source file: src/runtime/sequence.c
- documentation:

Missing:

Action: confirm skeleton ref handled identically to subject ref in sequence.

Last checked:

## ITEM-30 — Skeleton one-forward parity (§84)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation:

Missing: no one-forward capture for skeleton mode

Action: build skeleton fixture; compare to oracle IF oracle exposes it.

Last checked:

## ITEM-31 — Full skeleton-conditioned sanity (§76, §84)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected subject_skeleton.png
- source file:
- documentation:

Missing: no skeleton-conditioned image

Action: produce after semantics frozen.

Last checked:

## ITEM-32 — Full default recipe exact (50 steps, guidance 5.0, shift 3.0, UniPC) (§33, §85)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 9b03c14
- test: tests/unit/scheduler_matrix.c; make test-m17-base-local (E_local_18=9.25e-05)
- artifact: tests/unit/oracle_dump_matrix.txt
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing:

Action: confirm default UniPC parity in matrix test.

Last checked:

## ITEM-33 — Dev T2I recipe exact (§85)

Status: VERIFIED_DONE

Evidence:
- commit: c894178 feat(m1-post): derive dev recipe sigmas from frozen timesteps
- test: make test-m15 (test_m1_5_scheduler.c), make test-m17 (full dev gen)
- artifact: artifacts/m1/final_sanity/dev_native_seed123456.{png,json} (28/28 steps)
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing: none

Action: none

Last checked: 2026-09-17

## ITEM-34 — Guidance override exact for Full (§33, §85)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 9b03c14
- test: scheduler_matrix.c
- artifact:
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md (CFG)

Missing:

Action: confirm guidance override path.

Last checked:

## ITEM-35 — noise_scale_start exact (§34, §85)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 9b03c14
- test:
- artifact:
- source file: src/model/scheduler.c, src/runtime/generate.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md (Noise)

Missing:

Action: confirm noise_scale_start applied to initial noise (seed+1).

Last checked:

## ITEM-36 — noise_scale_end exact (§85)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing:

Action: confirm noise_scale_end usage.

Last checked:

## ITEM-37 — noise_clip_std exact (§85)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/model/scheduler.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing:

Action: confirm clip `noise_clip_std * noise.std()`.

Last checked:

## ITEM-38 — Unsupported combinations fail clearly (§52, §85)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/request.c (hd_request_validate), src/runtime/generate.c
- documentation: M1_POST_FULL_FEATURE_PARITY.md §52

Missing: error-path test not found

Action: targeted invalid-combination test.

Last checked:

## ITEM-39 — Per-step noise RNG semantics (§10, §34)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 131860d, 9b03c14
- test: make test-rng
- artifact: artifacts/m1/golden/M1_RNG
- source file: src/runtime/torch_rng.c
- documentation: docs/M1_POST_STATUS.md blocker 3

Missing: native stand-in (CPU MT19937) vs CUDA Philox documented, not parity-proven

Action: decide acceptable stand-in; record as documented divergence.

Last checked:

## ITEM-40 — Storyboard semantics identified from authoritative source (§36, §37, §64, §86)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 4f0a918, 801dd5a
- test:
- artifact: artifacts/m1post/audit/F_PROMPT_STORYBOARD_AUDIT.md
- source file:
- documentation: docs/M1_POST_MODE_CONTRACTS.md §6, capability matrix blocker 2

Missing: no storyboard/multi-panel decomposition found in frozen oracle source

Action: finalize audit in README/technical report/assets/examples; if unrecoverable, STOP and document ambiguity (§86).

Last checked:

## ITEM-41 — Storyboard native/orchestration implementation (§86, §38)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/request.h (HD_MODE_STORYBOARD), src/runtime/generate.c (rejects)
- documentation: M1_POST_FULL_FEATURE_PARITY.md §38

Missing: execution not implemented (generate.c rejects non-T2I)

Action: if semantics found, implement orchestration; else document upstream ambiguity.

Last checked:

## ITEM-42 — 3-panel deterministic storyboard sanity (§64, §86)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected storyboard_00/01/02.png
- source file:
- documentation:

Missing: no storyboard artifacts

Action: produce after ITEM-40/41.

Last checked:

## ITEM-43 — Prompt agent (legacy) response format supported (§40, §43, §87)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: artifacts/m1post/audit/F_PROMPT_STORYBOARD_AUDIT.md

Missing: no native prompt-agent client

Action: implement schema-compatible client testable with mocked endpoint.

Last checked:

## ITEM-44 — Dev-2604 Prompt-Refine response supported (§41, §87)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: docs/M1_POST_MODE_CONTRACTS.md §7

Missing: no native prompt-refiner client

Action: implement OpenAI-compatible client targeting localhost:8000/v1.

Last checked:

## ITEM-45 — raw/refined prompt boundary explicit + Python-free engine (§13, §42, §87)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/generate.c (no network/Python)
- documentation: M1_POST_FULL_FEATURE_PARITY.md §42, §73

Missing: boundary documented? confirmation needed

Action: confirm engine has no refiner dependency; document boundary.

Last checked:

## ITEM-46 — local-service/offline option documented (§42, §73, §87)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation:

Missing:

Action: document offline/no-refiner mode once ITEM-43/44 land.

Last checked:

## ITEM-47 — Progress callback contract (§45, §46, §71, §88)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 179d159/13be974 integration phase start (request.h)
- test:
- artifact:
- source file: src/runtime/request.h (hd_progress_callback), src/runtime/generate.c
- documentation: docs/M1_POST_MODE_CONTRACTS.md §8

Missing: callback unit test not found; per-step invocation unconfirmed

Action: callback unit test; confirm exact step count.

Last checked:

## ITEM-48 — Cancellation (§47, §88)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §47

Missing: cancellation support not confirmed

Action: verify/locate cancellation; targeted test.

Last checked:

## ITEM-49 — Preview callback + zero overhead when disabled (§48, §71, §88)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: docs/M1_POST_MODE_CONTRACTS.md §8 (previews at 1/4,1/2,3/4)

Missing: no preview extraction implementation

Action: implement optional preview; assert zero cost when disabled.

Last checked:

## ITEM-50 — Step count exact (§88)

Status: NEEDS_EVIDENCE

Evidence:
- commit: c894178
- test:
- artifact: artifacts/m1/final_sanity/dev_native_seed123456.json (28 steps)
- source file: src/runtime/generate.c
- documentation: docs/M1_POST_SCHEDULER_MATRIX.md

Missing:

Action: confirm 28/28 from artifact.

Last checked:

## ITEM-51 — CLI supports complete official surface (§50, §51, §89)

Status: PARTIAL

Evidence:
- commit: 4341f14 feat(m1-post): native image pipeline + generation CLI
- test:
- artifact:
- source file: src/main.c
- documentation: M1_POST_FULL_FEATURE_PARITY.md §50

Missing: --ref-image parsed? layout/skeleton JSON args? typed references; currently mode != T2I rejected downstream

Action: enumerate official CLI flags vs src/main.c; list gap; fix.

Last checked:

## ITEM-52 — CLI is a thin wrapper (§89)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/main.c (calls hd_generate)
- documentation:

Missing:

Action: confirm no logic duplication.

Last checked:

## ITEM-53 — Typed references/layout/skeleton in CLI (§89)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file: src/main.c (no layout/skeleton parsing)
- documentation:

Missing: CLI has no layout/skeleton/typed-ref args

Action: add after ITEM-51 gap list.

Last checked:

## ITEM-54 — Public C API supports same semantics (§53, §89)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 140ebdf, 448b412, 4341f14
- test:
- artifact:
- source file: include/hidream.h, src/runtime/request.h, src/runtime/generate.h
- documentation: M1_POST_FULL_FEATURE_PARITY.md §53

Missing: public header does not expose generation API (only profile/manifest)

Action: confirm public surface; decide whether generation must be in include/hidream.h.

Last checked:

## ITEM-55 — Mode validation / clear error paths (§52, §89)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file: src/runtime/request.c (hd_request_validate)
- documentation: M1_POST_FULL_FEATURE_PARITY.md §52

Missing:

Action: targeted validation test.

Last checked:

## ITEM-56 — Offline execution without Python (§26, §42)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 4341f14 (native C/CUDA only)
- test:
- artifact:
- source file: Makefile, src/
- documentation:

Missing: explicit offline audit not recorded

Action: confirm build/runtime has no Python dependency; document.

Last checked:

## ITEM-57 — Final sanity artifact matrix (§74)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected artifacts/m1_post/final_sanity/*.png (13 files)
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §74

Missing: 12 of 13 artifacts absent; only dev T2I 1024 exists (and under artifacts/m1/)

Action: produce each artifact as its item verifies.

Last checked:

## ITEM-58 — Sanity metadata JSON per artifact (§75)

Status: PARTIAL

Evidence:
- commit: 4341f14 (main.c writes .json sidecar)
- test:
- artifact: artifacts/m1/final_sanity/dev_native_seed123456.json
- source file: src/main.c
- documentation: M1_POST_FULL_FEATURE_PARITY.md §75

Missing: reference identifiers/hashes not populated for ref modes

Action: extend metadata writer for ref/layout/skeleton modes.

Last checked:

## ITEM-59 — Per-mode fixture set (§57)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected POST_T2I_1024 … POST_STORYBOARD
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §57

Missing: canonical fixture per mode not created

Action: create fixtures as modes verify.

Last checked:

## ITEM-60 — Mode-level validation ladder (§61)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §61

Missing: ladder (preproc/seq/1fwd/1step/3step/sanity) per mode not recorded per-mode

Action: record ladder results per mode.

Last checked:

## ITEM-61 — Full/Dev/Dev-2604 feature testing strategy (§62)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §62

Missing: Full profile sanity nowhere run; Dev-2604 modes not exercised

Action: after mode items land, run mandated matrix.

Last checked:

## ITEM-62 — M1-post/M2 boundary rule satisfied for every advertised mode (§69)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §69

Missing: modes still "unknown" (generate.c rejects all but T2I)

Action: resolve every mode to A/B/C.

Last checked:

## ITEM-63 — Performance specialization freeze table (§70)

Status: MISSING

Evidence:
- commit:
- test:
- artifact: expected sequence-length/attention/workspace table
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §70

Missing: no frozen real-sequence-profile table

Action: build after modes resolve.

Last checked:

## ITEM-64 — Sequence manifest diagnostics (§56)

Status: MISSING

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §56

Missing: manifest diagnostics not implemented

Action: determine scope; implement or document N/A.

Last checked:

## ITEM-65 — Web/API compatibility layer feasibility (§72)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §72

Missing: optional, must not block M2

Action: confirm optional; record NOT_APPLICABLE if undesired.

Last checked:

## ITEM-66 — Release sanity human-reviewable (§76)

Status: NEEDS_EVIDENCE

Evidence:
- commit: 4341f14
- test:
- artifact: artifacts/m1/final_sanity/dev_native_seed123456.png
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §76

Missing: only T2I visual; other modes lack artifacts

Action: resolve with ITEM-57.

Last checked:

## ITEM-67 — Required sub-milestone gates .2–.12 (§77–§90)

Status: PARTIAL

Evidence:
- commit: .0 c909300, .1 140ebdf..4341f14; .2–.12 no dedicated commits
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §77–§90

Missing: .2 through .12

Action: resolve via items above.

Last checked:

## ITEM-68 — Oracle capture policy compliance (§58, §59)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact: artifacts/m1post/audit/*
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §58, §59

Missing: run ledger entries for M1-post captures? oracle SHA recorded yes

Action: confirm captures recorded.

Last checked:

## ITEM-69 — Correctness levels V0–V6 reused (§60)

Status: NEEDS_EVIDENCE

Evidence:
- commit:
- test:
- artifact:
- source file:
- documentation: M1_POST_FULL_FEATURE_PARITY.md §60

Missing: per-mode level assignments not recorded

Action: record in ladder (ITEM-60).

Last checked:

---

## Remaining Work

### MUST FIX
- ITEM-09/10 long-text corpus + artifacts
- ITEM-14 keep-original-aspect
- ITEM-25/26 layout one-forward + sanity
- ITEM-30/31 skeleton one-forward + sanity
- ITEM-42 storyboard artifacts
- ITEM-43/44 prompt refiner clients
- ITEM-49 preview callback
- ITEM-53 typed refs/layout/skeleton CLI
- ITEM-57 final sanity artifact matrix
- ITEM-59 per-mode fixtures
- ITEM-63 performance freeze table
- ITEM-64 sequence manifest diagnostics

### NEEDS FINAL VALIDATION
- ITEM-01, 02, 03, 04, 05, 06, 07, 08, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 27, 28, 29, 32, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 60, 61, 62, 65, 66, 68, 69

### BLOCKED FOR M2
- (none yet — 2048 classification pending ITEM-06)

### COMPLETE
- ITEM-33 (1 / 69)
