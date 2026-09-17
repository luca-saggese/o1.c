# M1-post per-mode sequence fixtures (§57)

One canonical fixture per execution mode. These are **sequence-level
manifests**: they freeze the mode, profile, prompt, dims, seed, steps,
scheduler, expected total sequence length S and reference-token count.
They do **not** include generated images — GPU runs are serialized by
the lead; fixtures that require a GPU oracle capture are marked
`status: pending_gpu` and listed under Pending below.

All S / ref-token numbers are computed by the unified sequence builder
(`src/runtime/sequence.c`) and frozen in
`docs/M1_POST_PERF_FREEZE.md` (§70). Model dims: H=4096, NH=32, NKV=8,
HD=128, ff_hidden=12288, head_out=3072, NLAYERS=36. PATCH=32,
FIX_POINT=4096, TIMESTEP_TOKENS=1, SPATIAL_MERGE=1.

## Fixtures

| manifest | mode | profile | expected S | ref tokens | status |
|---|---|---|---|---|---|
| `POST_T2I_1024.json` | t2i | T2I 1024 | 1043 | 0 | ready |
| `POST_T2I_2048.json` | t2i | T2I 2048 | 4115 | 0 | ready |
| `POST_T2I_LONG_TEXT.json` | t2i | T2I 1024 (long prompt) | 1082 | 0 | ready |
| `POST_EDIT_1REF.json` | edit | edit 1ref | 1440 | 256 | ready |
| `POST_EDIT_1REF_KEEP_ASPECT.json` | edit | edit 1ref keep-aspect | 4240 | 2048 | ready |
| `POST_SUBJECT_2REF.json` | personalize | subject 2ref | 1619 | 288 | ready |
| `POST_SUBJECT_MULTIREF.json` | personalize | subject many-ref | 2855 | 1152 | ready |
| `POST_SUBJECT_LAYOUT.json` | layout | layout | 2245 | 768 | ready |
| `POST_SUBJECT_SKELETON.json` | skeleton | skeleton | 2648 | 1024 | ready |
| `POST_STORYBOARD.json` | t2i | storyboard | 1082 | 0 | ready |

## Sources

- **T2I fixtures**: native sequence build via `hd_seq_t2i`
  (`src/runtime/sequence.c`), canonical M1.4 golden input_ids
  (19 tokens) for the 1024/2048 fixtures; long-text prompt tokenized
  with `hd_tokenizer_encode_prompt` + boi + tms for
  `POST_T2I_LONG_TEXT`.
- **Edit / personalize / layout / skeleton fixtures**: native sequence
  build via `hd_seq_build` with oracle ref geometry
  (`python/models/pipeline.py` ref path + `python/models/utils.py`
  `calculate_dimensions`), verified structurally by
  `tests/unit/test_seq_ref.c`.
- **Storyboard**: multi-panel single-pass T2I from a long sequential
  prompt — no dedicated code path
  (`docs/M1_POST_STORYBOARD_FINAL_AUDIT.md`). Represented as a
  long-text T2I sequence (3-panel prompt, 1024×1024 panels).

## Pending

None. All manifests are sequence-level and CPU-computable; no GPU
oracle capture is required for the manifest itself. GPU runs that
consume these fixtures (image generation, oracle byte-capture) are
serialized by the lead and are out of scope for the manifests.

## Scheduler recipes

From `docs/M1_POST_SCHEDULER_MATRIX.md`:

- **Dev**: 28 steps, guidance 0.0, shift 1.0, flash, DEFAULT_TIMESTEPS
  (T2I 1024, T2I 2048, T2I long text, storyboard).
- **Dev edit**: 28 steps, guidance 0.0, shift 1.0, flow_match/flash,
  DEFAULT_TIMESTEPS (edit 1ref, edit 1ref keep-aspect, subject 2ref,
  subject many-ref, layout, skeleton).

`DEFAULT_TIMESTEPS = [999, 987, 974, 960, 945, 929, 913, 895, 877, 857,
836, 814, 790, 764, 737, 707, 675, 640, 602, 560, 515, 464, 409, 347,
278, 199, 110, 8]`, NOISE_SCALE = 8.0, CONDITION_IMAGE_SIZE = 384.