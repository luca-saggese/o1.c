# M1-post performance / shape freeze (§70)

Frozen sequence geometry for every real sequence profile. Numbers are
computed by the unified sequence builder (`src/runtime/sequence.c`) via
`tests/unit/seq_profiles.c` (`make test-seq-profiles`), except the T2I
2048 row which is frozen from the BLOCKED_M2 DEV-2048 run
(`docs/M1_POST_CLOSEOUT_AUDIT.md` ITEM-06).

Model dims (frozen, `tests/unit/sanity_gen.c`): H=4096, NH=32, NKV=8,
HD=128, ff_hidden=12288, head_out=3072, NLAYERS=36. PATCH=32,
FIX_POINT=4096, TIMESTEP_TOKENS=1, SPATIAL_MERGE=1.

Workspace = documented estimate (persistent buffers + block scratch,
see `hd_seq_diag` in `src/runtime/sequence.c`): attention scores+probs
alone are 2·NH·S² bf16 = 128·S² bytes (reproduces ITEM-06: S=4115 →
2,167,452,800 B = 2165.9 MB). The T2I 2048 total (3,056,535,432 B ≈
2.85 GB) matches the ITEM-06 peak 2.7–2.9 GB.

| profile | total S | attention shape | workspace bytes | ref tokens | target tokens | scheduler recipe |
|---|---|---|---|---|---|---|
| T2I 1024 | 1043 | 1043×1043 | 364,578,696 | 0 | 1024 | Dev: 28 steps, guidance 0.0, shift 1.0, flash, DEFAULT_TIMESTEPS |
| T2I 2048 | 4115 | 4115×4115 | 3,056,535,432 (scores+probs 2,167,452,800 = 2165.9 MB) | 0 | 4096 | Dev: 28 steps, guidance 0.0, shift 1.0, flash, DEFAULT_TIMESTEPS |
| edit 1ref | 1440 | 1440×1440 | 573,622,024 | 256 | 1024 | Dev edit: 28 steps, guidance 0.0, shift 1.0, flow_match/flash, DEFAULT_TIMESTEPS |
| edit 1ref keep-aspect | 4240 | 4240×4240 | 3,195,995,912 | 2048 | 2048 | Dev edit: 28 steps, guidance 0.0, shift 1.0, flow_match/flash, DEFAULT_TIMESTEPS |
| subject 2ref | 1619 | 1619×1619 | 681,756,552 | 288 | 1024 | Dev edit: 28 steps, guidance 0.0, shift 1.0, flow_match/flash, DEFAULT_TIMESTEPS |
| subject many-ref | 2855 | 2855×2855 | 1,647,025,032 | 1152 | 1024 | Dev edit: 28 steps, guidance 0.0, shift 1.0, flow_match/flash, DEFAULT_TIMESTEPS |
| layout | 2245 | 2245×2245 | 1,121,413,000 | 768 | 1024 | Dev edit: 28 steps, guidance 0.0, shift 1.0, flow_match/flash, DEFAULT_TIMESTEPS |
| skeleton | 2648 | 2648×2648 | 1,457,964,808 | 1024 | 1024 | Dev edit: 28 steps, guidance 0.0, shift 1.0, flow_match/flash, DEFAULT_TIMESTEPS |
| storyboard | 1082 | 1082×1082 | 383,533,320 | 0 | 1024 | Dev: 28 steps, guidance 0.0, shift 1.0, flash, DEFAULT_TIMESTEPS |

## Profile definitions

- **T2I 1024 / T2I 2048**: text-to-image, canonical 19-token prompt
  (M1.4 golden input_ids), 1024×1024 / 2048×2048 target. S = text_len +
  (H/32)·(W/32) = 19 + 1024 / 19 + 4096.
- **edit 1ref**: 1024×1024 target, one 512×512 reference (16×16 grid =
  256 tokens), VLM cond grid 12×12. S = text_len + 1024 + 256.
- **edit 1ref keep-aspect**: portrait 1024×2048 target derived from a
  1024×2048 reference (32×64 grid = 2048 tokens), cond grid 8×16.
- **subject 2ref**: 1024×1024 target, two 384×384 references (12×12 =
  144 tokens each), cond 12×12 each.
- **subject many-ref**: K=8, 1024×1024 target, eight 384×384 references
  (144 tokens each), cond 9×9 each (cond_img_size 288).
- **layout**: 2 subject refs + 1 layout canvas = K=3, refs 512×512
  (256 tokens each), cond 12×12 each.
- **skeleton**: subject photo + 3 poses = K=4, refs 512×512 (256 tokens
  each), cond 12×12 each.
- **storyboard**: multi-panel single-pass T2I from a long sequential
  prompt (no dedicated code path, `M1_POST_STORYBOARD_FINAL_AUDIT.md`);
  representative 3-panel prompt, 1024×1024 panels. S = text_len + 1024.

Scheduler recipes from `docs/M1_POST_SCHEDULER_MATRIX.md`:
`DEFAULT_TIMESTEPS = [999, 987, 974, 960, 945, 929, 913, 895, 877, 857,
836, 814, 790, 764, 737, 707, 675, 640, 602, 560, 515, 464, 409, 347,
278, 199, 110, 8]`, NOISE_SCALE = 8.0, CONDITION_IMAGE_SIZE = 384.