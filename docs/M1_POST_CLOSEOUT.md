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
| native RNG | DONE | 131860d | make test-rng PASS; artifacts/m1/golden/M1_RNG | — | A |
| 1024² generation | DONE | b17cbcc | make test-sanity; dev_native_seed123456.png 1024×1024 | — | A |
| 2048 | BLOCKED_M2 | 140ebdf (shape-generic) | ITEM-06: S=4115, workspace ≈2.17GB (peak 2.7–2.9GB), failure at materialized O(S²) attention | measured blocker, evidence in audit ITEM-06 | A |
| long text / multilingual | DONE | 94501b1 | make test-tokenizer PASS; artifacts/m1_post/final_sanity/long_text_{en,zh}.{png,json} | — | A |
| single-ref editing | DONE | 2bc625e | make test-seq-ref PASS (K=1/K=2, 55 checks) | end-to-end edit run gated in generate.c (execution, not sequence) | B |
| keep-original-aspect | MISSING | — | request.h field only | no handling in sequence.c/generate.c; edit_keep_aspect.png missing | B |
| Dev edit flow_match | NEEDS_CHECK | 9b03c14 | scheduler_matrix.c flow_match parity vs oracle | no end-to-end edit run; generate.c rejects non-flash | B |
| Dev edit flash | NEEDS_CHECK | c894178, 9b03c14 | scheduler_matrix.c flash parity | edit-mode end-to-end gated | B |
| Full edit | MISSING | — | — | no Full edit implementation; generate.c rejects all edit modes | B |
| Full scheduler/guidance | NEEDS_CHECK | 9b03c14 | matrix UniPC 3-step parity; defaults 50/5.0/3.0 in request.h | no Makefile target; no 50-step run | B |
| noise controls | NEEDS_CHECK | 9b03c14 | applied in generate.c:344,390-415; matrix tests noise_scale_schedule | no dedicated test/artifact | B |
| multi-ref personalization | DONE | 2bc625e | make test-seq-ref PASS (K=1/K=2, ordered concat) | subject_2ref.png artifact; uncommitted partial preprocess in generate.c | C |
| many-ref handling | NEEDS_CHECK | 2bc625e | hd_seq_build count-generic, HD_SEQ_MAX_REFS bound | no K>2 stress test/artifact | C |
| layout conditioning | DONE | 5cd4134 | make test-layout PASS (bit-exact, max abs diff 0.000000) | one-forward parity + subject_layout.png | C |
| skeleton conditioning | MISSING | 4f0a918 | HD_REF_SKELETON defined; no role-specific branch in hd_seq_build | no skeleton test/one-forward/artifact; role semantics unresolved | C |
| storyboard | MISSING | 4f0a918, 801dd5a | audit F (no storyboard in oracle) | semantics unrecovered; no orchestration; no 3-panel artifacts | D |
| prompt agent/refiner | MISSING | — | audit F (contracts only) | no native client (legacy + Dev-2604) | D |
| progress callback | NEEDS_CHECK | 179d159/13be974 | invoked once per step generate.c:424 | no unit test; signature lacks get_preview arg | D |
| preview callback | MISSING | — | — | no preview extraction; zero-overhead-when-disabled | D |
| cancellation | NOT_REQUIRED | — | spec §47: robust cancellation not exposed/required | — | D |
| CLI | MISSING | 4341f14 | build/hidream --help | --ref-image NOT parsed; missing --model-path, --guidance-scale, --noise-scale-start/end, --noise-clip-std, --editing-scheduler, --keep-original-aspect, --layout-bboxes | E |
| public C API | MISSING | 140ebdf, 448b412 | include/hidream.h (profile/manifest only) | generation API not in public header | E |
| offline/Python-free runtime | DONE | 4341f14 | Makefile C/CUDA only; no Python/network/HTTP in src/ | — | E |
| invalid-combination handling | NEEDS_CHECK | — | request.c hd_request_validate (profile/dims/steps/refs) | no dedicated validation test | E |
| final sanity artifacts | MISSING | — | only long_text_{en,zh} + dev T2I exist | 11 of 13 §74 artifacts missing | F |
| per-mode fixtures | DONE | 83a1fbb | fixtures/m1_post/ 10 POST_* manifests, all status:ready, text_len cross-checked | — | F |
| perf freeze table | DONE | 8129b62 | make test-seq-profiles PASS; docs/M1_POST_PERF_FREEZE.md §70 (S/ws per mode) | — | F |
| sequence manifest diagnostics | DONE | e2f2aea | make test-seq-diag PASS (7/7); hd_seq_diag prints §56 manifest + workspace estimate | — | F |

## In-progress / uncommitted (from interrupted session)

- `src/runtime/generate.c`: partial multi-ref preprocessing helpers
  (`preprocess_ref`, `ref_cond_grid`) — NOT wired into hd_generate, NOT committed.
  Block C must decide: complete + commit, or discard.
- `docs/M1_POST_CLOSEOUT_AUDIT.md`: uncommitted evidence updates (ITEM-11..14).

## Remaining Work (after block audit A–F, 2026-09-17)

### MUST FIX
- keep-original-aspect (ITEM-14) — no handling in sequence.c/generate.c
- Full edit (ITEM-17) — generate.c rejects all edit modes
- skeleton conditioning (ITEM-27..31) — no role branch, no test/artifact
- storyboard (ITEM-40..42) — semantics unrecovered; needs final upstream decision
- prompt agent/refiner (ITEM-43/44) — no native clients
- preview callback (ITEM-49) — no implementation
- CLI (ITEM-51/53) — --ref-image not parsed; 8 parity flags missing
- public C API (ITEM-54) — generation not exposed in include/hidream.h
- final sanity artifacts (ITEM-57) — 11 of 13 missing

### NEEDS FINAL VALIDATION
- Dev edit flow_match/flash end-to-end (ITEM-15/16)
- Full scheduler/guidance 50-step (ITEM-32/34)
- noise controls dedicated test (ITEM-35..37)
- many-ref K>2 stress (ITEM-20)
- progress callback unit test (ITEM-47)
- invalid-combination validation test (ITEM-38/55)

### BLOCKED FOR M2
- 2048 (ITEM-06): S=4115, workspace ≈2.17GB, O(S²) attention

### COMPLETE
- T2I, RNG, 1024, long-text, single-ref seq, multi-ref seq, layout parser,
  offline runtime, capability matrix, per-mode fixtures, perf freeze table,
  sequence manifest diagnostics