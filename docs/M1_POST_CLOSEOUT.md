# M1-post Closeout

Source of truth for M1-post closure. Detailed per-item evidence lives in
`docs/M1_POST_CLOSEOUT_AUDIT.md`; this file is the feature-level operational
table. Old tasklist is NOT authoritative.

Engine HEAD: 73488bc
Oracle SHA: 3237a638a5c2c7be106b0175958f4c0db8c2dfbf
Full/Base revision: 0b0901d99f200389e138c61946af1185f5f49a13
Dev revision: b6acc2fe452b3120430620dc4354fa442ee081ea
Audit started: 2026-09-17

## Status legend

- DONE — commit + test/artifact evidence
- NEEDS_CHECK — evidence incomplete, targeted check required
- MISSING — no implementation/evidence
- BLOCKED_M2 — semantics correct, blocked purely by perf/memory
- UNSUPPORTED_UPSTREAM — frozen upstream proves profile/feature unsupported

## Feature table

| feature | stato | commit | test/artifact | cosa manca | owner |
|---------|-------|--------|---------------|------------|-------|
| production T2I | DONE | 140ebdf, b17cbcc, 4341f14 | artifacts/m1/final_sanity/dev_native_seed123456.{png,json,log} (28/28 steps, 5/5 asserts) | — | A |
| native RNG | DONE | 131860d | make test-rng; artifacts/m1/golden/M1_RNG | — | A |
| 1024² generation | DONE | b17cbcc | make test-sanity; dev_native_seed123456.png 1024×1024 | — | A |
| 2048 | BLOCKED_M2 | — | ITEM-06: S=1077 workspace 381MB; 2048² → S≈4101, O(S²) attention OOM | measured blocker, see audit ITEM-06 | A |
| long text / multilingual | DONE | 94501b1 | make test-tokenizer; artifacts/m1_post/final_sanity/long_text_{en,zh}.{png,json} | — | A |
| single-ref editing | NEEDS_CHECK | 2bc625e | make test-seq-ref (K=1 parity) | end-to-end edit run gated in generate.c | B |
| keep-original-aspect | PARTIAL | — | request.h field only | behavior not wired/verified; artifact edit_keep_aspect.png missing | B |
| Dev edit flow_match | NEEDS_CHECK | 9b03c14 | tests/unit/scheduler_matrix.c | generate.c rejects non-flash scheduler | B |
| Dev edit flash | NEEDS_CHECK | c894178, 9b03c14 | scheduler_matrix.c | edit-mode end-to-end gated | B |
| Full edit | NEEDS_CHECK | — | — | Full profile edit run not found | B |
| Full scheduler/guidance | NEEDS_CHECK | 9b03c14 | scheduler_matrix.c; make test-m17-base-local | confirm UniPC parity + guidance override | B |
| noise controls | NEEDS_CHECK | 9b03c14 | scheduler_matrix.c | confirm start/end/clip exact | B |
| multi-ref personalization | NEEDS_CHECK | 2bc625e | make test-seq-ref (K=2) | many-ref stress; subject_2ref.png artifact; uncommitted partial preprocess in generate.c | C |
| layout conditioning | NEEDS_CHECK | 5cd4134 | make test-layout ALL PASS (bit-exact) | one-forward parity + subject_layout.png | C |
| skeleton conditioning | NEEDS_CHECK | 4f0a918 | audit E; seq-ref path | role semantics unresolved; one-forward + subject_skeleton.png | C |
| storyboard | NEEDS_CHECK | 4f0a918, 801dd5a | audit F (no storyboard in oracle) | semantics unrecovered; 3-panel artifacts | D |
| prompt agent/refiner | MISSING | — | audit F (contracts only) | no native client (legacy + Dev-2604) | D |
| progress callback | NEEDS_CHECK | 179d159/13be974 | request.h hd_progress_callback | per-step invocation test; cancellation | D |
| preview callback | MISSING | — | — | no preview extraction; zero-overhead-when-disabled | D |
| CLI | PARTIAL | 4341f14 | build/hidream --help | no --ref-image parse, no layout/skeleton args, non-T2I rejected | E |
| public C API | NEEDS_CHECK | 140ebdf, 448b412 | include/hidream.h + runtime headers | generation API not in public header | E |
| offline/Python-free runtime | NEEDS_CHECK | 4341f14 | native C/CUDA build | explicit offline audit not recorded | E |
| invalid-combination handling | NEEDS_CHECK | — | request.c hd_request_validate | error-path test missing | E |
| final sanity artifacts | MISSING | — | only long_text_{en,zh} + dev T2I exist | 11 of 13 §74 artifacts missing | F |
| per-mode fixtures | MISSING | — | — | POST_* fixture set not created | F |
| perf freeze table | MISSING | — | — | §70 sequence/attention/workspace table | F |
| sequence manifest diagnostics | MISSING | — | — | §56 diagnostics not implemented | F |

## In-progress / uncommitted (from interrupted session)

- `src/runtime/generate.c`: partial multi-ref preprocessing helpers
  (`preprocess_ref`, `ref_cond_grid`) — NOT wired into hd_generate, NOT committed.
  Block C must decide: complete + commit, or discard.
- `docs/M1_POST_CLOSEOUT_AUDIT.md`: uncommitted evidence updates (ITEM-11..14).

## Remaining Work (initial)

### MUST FIX
- keep-original-aspect (ITEM-14)
- prompt agent/refiner clients (ITEM-43/44)
- preview callback (ITEM-49)
- CLI typed refs/layout/skeleton (ITEM-53)
- final sanity artifact matrix (ITEM-57)
- per-mode fixtures (ITEM-59)
- perf freeze table (ITEM-63)
- sequence manifest diagnostics (ITEM-64)

### NEEDS FINAL VALIDATION
- editing end-to-end, schedulers, multi-ref, skeleton, storyboard, progress,
  C API, offline audit, invalid combos

### BLOCKED FOR M2
- 2048 (ITEM-06)

### COMPLETE
- T2I, RNG, 1024, long-text, layout parser, capability matrix