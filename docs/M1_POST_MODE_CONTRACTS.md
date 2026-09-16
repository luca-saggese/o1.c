# M1-post Mode Contracts

Frozen at M1-post.0. Each mode's exact sequence/conditioning contract, from the
official upstream audit. Source: `python/models/pipeline.py`, `utils.py`,
`inference.py`, `app.py` (frozen oracle).

## 1. T2I

- Sequence: `build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config)` (pipeline.py:31-77).
  - chat template (user role) + `<|boi_token|>` + `<|tms_token|>`×1 → tokenize (no special tokens).
  - append `image_len` vision tokens: first = `vision_start_token_id` (151652), rest = `image_token_id` (151655).
  - `image_len = (H//32) * (W//32)`.
- Position IDs: `get_rope_index_fix_point(..., fix_point=4096)` — text rows 0..text_len-1, image rows 4096.
- token_types: TMS row = 3, image rows = 1, text rows = 0; binarized for mask.
- vinput_mask = (token_types == 1).
- Causal mask: text rows causal (triu min_val), TMS+image rows all-zero.
- CFG: when guidance > 1.0, second pass with blank `" "` prompt (pipeline.py:161-165).
- Output: unpatchify `(z+1)/2`, clamp, round, uint8 RGB. No VAE.

## 2. Single-reference editing

- Exactly one reference image (pipeline.py:170-251).
- Reference: `Image.open().convert("RGB")`, `resize_pilimage` (BICUBIC, patch-aligned, center-crop), normalize [-1,1], patchify 32×32.
- `keep_original_aspect` (only with 1 ref): resize ref to max_size=2048, derive target W/H from ref (pipeline.py:141-157).
- Sequence: one target token sequence + one image sequence per reference.
- Dev scheduler: flow_match default, flash optional.

## 3. Multi-reference personalization

- >= 2 references, ordered concat (pipeline.py:213-234).
- Reference max sizing by count: K=1 full, K=2 ¾, K≤4 ½, K≤8 ⅜, else ¼.
- No role tags; per-ref captions unsupported.

## 4. Layout conditioning

- JSON: list or wrapper keys (`layout_bboxes`/`bboxes`/`boxes`/`bbox_list`).
- Item: `[x1,x2,y1,y2]` or `{bbox/box, text/label}`; coords normalized [0,1] or percent [0,100].
- Rendered as black RGB canvas (H,W,3) with up to 5 colored outlined boxes, appended as one extra reference image (utils.py:130-185).
- NOT an attention-mask tensor.

## 5. Skeleton conditioning

- **SUPPORTED_BY_UPSTREAM / IMPLEMENTATION_REQUIRED** (see capability matrix).
- The official upstream `main` declares IP-pipeline skeleton support and ships the
  "Multi-Reference Subject-Driven Personalization with Skeleton" example, using
  `face`, `background`, `openpose`, and part references through the normal
  `--ref_images` path. No dedicated OpenPose parser exists in the frozen oracle
  source; skeleton semantics are carried by the generic multi-reference path.
- Native engine: route skeleton refs through the same `hd_seq_build` multi-ref
  path (K refs, ordered concat), with the skeleton image as one of the refs.
  Whether face/bg/openpose/parts are differentiated by order, metadata, filename,
  or content only is still under audit (see capability matrix blockers).

## 6. Storyboard

- **ADVERTISED_CAPABILITY / SEMANTICS_NOT_YET_ESTABLISHED** (see capability matrix).
- The official README advertises storyboard as a model feature, but no dedicated
  API/pipeline was found in the frozen oracle. Audit continues in README,
  technical report, assets/examples, and prompt-agent/web workflow before
  deciding how to implement it. Do not close as unsupported merely because no
  flag/string with that name exists in the model code.

## 7. Prompt refinement

- Legacy: OpenAI-compatible chat.completions, system prompt at prompt_agent.py:6-64, user = raw input.
- Dev-2604: prompt_agent_v2.py:7-43, English rewrite only, targets `HiDream-ai/Prompt-Refine` @ localhost:8000/v1.

## 8. Progress / previews

- `callback(step_idx, total, get_preview)` once per step (pipeline.py:409-429).
- Previews at 1/4, 1/2, 3/4 milestones, downscaled JPEG ≤384px (app.py:801-846).

## 9. Output decode (all modes)

- Pixel-space unpatchification; `(z+1)/2`; clamp [0,255]; round; uint8 RGB. No VAE/decoder.