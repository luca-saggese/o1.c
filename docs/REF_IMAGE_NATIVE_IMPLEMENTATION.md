# Ref-Image Native Implementation

Persistent memory for the reference-image visual-conditioning task in o1.c.
Every new discovery MUST be appended here immediately. Do not rely on
conversation history.

Last updated: (stage A start)

## FROZEN FACTS

- `ref_patches -> vinputs` path is ALREADY implemented and works (native HAS
  path 1 of upstream dual-path conditioning).
- The native forward is MISSING the entire Qwen3-VL visual tower path:
  ref image -> VLM preprocessing -> vision tower -> merged image embeddings
  -> image placeholder replacement -> deepstack injection.
- Without path 2, ref-mode generations produce gray output (x_pred ~= -0.03,
  decode -> ~0.485 gray) because the decoder sees placeholder embeddings for
  `<image>` tokens instead of real visual features.
- Scheduler is CORRECT. Do NOT touch scheduler. Gray output = wrong forward.

## Upstream files / functions (canonical reference)

- `_reference/HiDream-O1-Image/models/pipeline.py`
  - ref path: lines ~160-300 (max_size, resize_pilimage, ref_patches,
    cond_img_size, ref_pils_vlm, image_grid_thw_ref, processor call,
    input_ids_pad, position_ids, token_types, vinput_mask)
  - `forward_once` (line ~333): kwargs include pixel_values, image_grid_thw,
    precomputed_image_embeds, precomputed_deepstack_image_embeds
  - denoise loop (line ~369): vinputs = cat([z, ref_patches]); first step
    computes image_embeds + deepstack, caches them, reuses for later steps
  - `vinputs = torch.cat([z, ref_patches], dim=1)` (line ~388)
- `_reference/HiDream-O1-Image/models/qwen3_vl_transformers.py`
  - `Qwen3VLModel._forward_generation` (line ~1400): THE reference for the
    native forward. Steps:
    1. inputs_embeds = embed_tokens(input_ids)
    2. if pixel_values: image_embeds, deepstack = get_image_features();
       image_mask = get_placeholder_mask(); inputs_embeds =
       inputs_embeds.masked_scatter(image_mask, image_embeds)
    3. t_emb = t_embedder1(timestep); tms_mask replace
    4. vinputs_embedded = x_embedder(vinputs); cat([inputs_embeds, vemb])
    5. visual_pos_masks padded with False for vinputs portion
    6. decoder with deepstack injection at layers 8/16/24
    7. final_layer2 -> x_pred
  - `Qwen3VLModel.forward` (line ~1625): standard path, same masked_scatter
  - `Qwen3VLVisionModel.forward` (line ~728): patch_embed -> +pos_embeds ->
    rot_pos_emb -> 27 blocks -> merger; deepstack mergers at layers 8/16/24
  - `Qwen3VLVisionBlock.forward` (line ~436): norm1 -> attn -> residual ->
    norm2 -> mlp -> residual
  - `Qwen3VLVisionAttention.forward` (line ~197): qkv -> rotary -> eager attn
    (per-image chunks via cu_seqlens) -> proj
  - `Qwen3VLVisionPatchMerger.forward` (line ~118): norm -> fc1 -> GELU ->
    fc2
  - `_deepstack_process` (line ~938): hidden_states[visual_pos_masks] +=
    visual_embeds
  - `get_placeholder_mask` (line ~1220): input_ids == image_token_id
  - `fast_pos_embed_interpolate` (line ~660): 4-corner bilinear interp of
    pos_embed over grid_h/grid_w
  - `rot_pos_emb` (line ~630): rotary freq table lookup by (row,col) coords
  - `Qwen3VLVisionRotaryEmbedding` (line ~102): inv_freq = 1/(theta^(arange/2/dim))
  - `apply_rotary_pos_emb_vision` (line ~140): q*cos + rotate_half(q)*sin
- `_reference/HiDream-O1-Image/models/utils.py`
  - `calculate_dimensions(max_size, ratio)` (line ~239): width =
    sqrt(max_size^2 * ratio); /32 aligned
  - `resize_pilimage`, `create_layout_reference_images`, `load_layout_bboxes`

## Sequence layout (native hd_seq_build, ref mode)

- `[text_len][target image_len][ref blocks ref_len[i]]`
- text_len = template (with K expanded `<image_pad>` cond grids) + boi + tms
- target rows: `out_dev[text_len : text_len+IMG]`
- token types: 1=target, 2=refs, 3=tms
- mask zeroes token rows (gen tokens attend to everything)
- input_ids: text stream contains `<vision_start>` + cond_grid `<image_pad>`
  tokens (151652 + 151655*cond_h*cond_w) per ref, then boi(151669) + tms
- vision blocks appended after text: tgt block then ref blocks
- `hd_seq_build(req, PATCH, 151655, 151656, 151652, TMS_ID, 1, 1, 4096, Hh, W, refs, &seq)`
- TMS_ID = 151673, PATCH = 32, FF = 3072, CONDITION_IMAGE_SIZE = 384

## ref_patches path (native, DONE)

- generate.c `hd_generate_ref()`: loads refs, resizes to max_size (K==1:
  max(h,w); K==2: max*48/64; K<=4: max/2; K<=8: max*24/64; else max/4),
  patch-aligns, normalizes [-1,1], pixel_unshuffles to [tokens, 3072]
- refs[r].tokens = gh*gw (full-res patches), refs[r].cond_h/cond_w = VLM
  cond grid (post spatial_merge)
- vinputs = cat([z, ref_patches]) per step, uploaded as bf16
- forward selects target rows text_len..text_len+IMG

## pixel_values path (native, MISSING)

Upstream processor (Qwen2VLImageProcessor, transformers 4.57.1 in .venv):
- input: ref_pils_vlm = each resized ref resized to cond dims
  (calculate_dimensions(384, ratio), LANCZOS)
- smart_resize(factor=patch_size*merge_size=32, min_pixels=65536,
  max_pixels=16777216) -> resized to (resized_h, resized_w)
- rescale 1/255, normalize mean=0.5 std=0.5 -> [-1,1]
- patchify: reshape to (grid_t, t, C, grid_h//m, m, p, grid_w//m, m, p)
  transpose -> flatten to [grid_t*grid_h*grid_w, C*t*p*p]
- output pixel_values shape [520, 1536] for test.jpg (grid [1,26,20])
- NOTE: pixel_values is the PATCHIFIED input (already [tokens, patch_dim]),
  NOT a raw image. patch_dim = 3*2*16*16 = 1536.
- image_grid_thw = [1, 26, 20] (grid_h, grid_w in patches of the cond image)

## image_grid_thw

- ref: `image_grid_thw_ref[i] = [1, rh//PATCH, rw//PATCH]` where rh/rw are the
  resized ref dims (max_size-resized, NOT cond dims)
- tgt: `[1, height//PATCH, width//PATCH]`
- cond (processor): `proc.image_grid_thw` = cond grid (post-merge divided by
  spatial_merge_size in pipeline: `igthw_cond[i,1] //= 2`)
- vision tower uses grid_thw = [1, grid_h, grid_w] in patches (26, 20 for
  test.jpg)

## Vision tower stages (upstream, exact)

1. patch_embed: Conv3d(3, 1152, kernel=(2,16,16), stride same, bias)
   - weight [1152, 3, 2, 16, 16], bias [1152]
   - input pixel_values [520, 1536] -> view(-1, 3, 2, 16, 16) -> conv ->
     [520, 1152]
   - NOTE: pixel_values already patchified; conv is a matmul over the
     flattened patch dim (1536 -> 1152)
2. + pos_embeds: fast_pos_embed_interpolate(grid_thw) -> [520, 1152]
   - 4-corner bilinear interp of pos_embed.weight [2304, 1152] over
     grid_h/grid_w, then spatial-merge permute (view(t, h//m, m, w//m, m, -1)
     .permute(0,1,3,2,4,5).flatten(0,4))
3. rot_pos_emb: rotary freq table [520, 1152] (dim 1152, head_dim 72,
   rotary dim 36 -> inv_freq [18])
   - emb = cat([rot, rot], -1); position_embeddings = (cos, sin)
4. 27 blocks (Qwen3VLVisionBlock):
   - norm1 (LayerNorm 1152, eps 1e-6) -> attn -> residual
   - attn: qkv Linear(1152, 3456) -> reshape (seq, 3, 16, 72) -> q/k/v
     [16, seq, 72]; rotary; eager attention per-image (cu_seqlens chunks,
     no mask, scaling = head_dim^-0.5); proj Linear(1152, 1152)
   - norm2 -> mlp (fc1 Linear(1152, 4304) GELU fc2 Linear(4304, 1152)) ->
     residual
5. deepstack mergers at layers 8, 16, 24:
   - Qwen3VLVisionPatchMerger(use_postshuffle_norm=True): norm over
     hidden_size*4 = 4608 (post-shuffle), fc1 Linear(4608, 4608), GELU,
     fc2 Linear(4608, 4096)
   - input: block output [520, 1152] -> spatial merge (2x2 unshuffle) ->
     [130, 4608] -> norm -> fc1 -> gelu -> fc2 -> [130, 4096]
6. merger (final): Qwen3VLVisionPatchMerger(use_postshuffle_norm=False):
   - norm over 1152, fc1 Linear(4608, 4608), GELU, fc2 Linear(4608, 4096)
   - input: block 26 output [520, 1152] -> spatial merge -> [130, 4608] ->
     norm(1152) -> fc1 -> gelu -> fc2 -> [130, 4096]
7. image_embeds = [130, 4096] (cat of per-image splits)
8. deepstack_image_embeds = 3 x [130, 4096]

## image_embeds / placeholder replacement

- image_embeds [130, 4096] bf16
- image_mask = (input_ids == 151655) expanded to [1, seq, 4096]
- inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
  -> replaces the 130 `<image_pad>` positions in the text stream with the
  visual features (in order)
- deepstack: visual_pos_masks = image_mask[...,0]; deepstack_visual_embeds =
  deepstack_image_embeds (3 x [130, 4096]); injected at decoder layers 8/16/24
  via `hidden_states[visual_pos_masks] += deepstack_embeds[layer_idx]`
- visual_pos_masks padded with False for the vinputs portion (target+refs)

## Relevant weight names / shapes (from config/tensor_manifest_dev.json)

- model.visual.patch_embed.proj.weight [1152, 3, 2, 16, 16] (f32 in manifest)
- model.visual.patch_embed.proj.bias [1152]
- model.visual.pos_embed.weight [2304, 1152]
- model.visual.blocks.{0..26}.norm1.weight/bias [1152]
- model.visual.blocks.{0..26}.norm2.weight/bias [1152]
- model.visual.blocks.{0..26}.attn.qkv.weight [3456, 1152], qkv.bias [3456]
- model.visual.blocks.{0..26}.attn.proj.weight [1152, 1152], proj.bias [1152]
- model.visual.blocks.{0..26}.mlp.linear_fc1.weight [4304, 1152], bias [4304]
- model.visual.blocks.{0..26}.mlp.linear_fc2.weight [1152, 4304], bias [1152]
- model.visual.merger.norm.weight/bias [1152]
- model.visual.merger.linear_fc1.weight [4608, 4608], bias [4608]
- model.visual.merger.linear_fc2.weight [4096, 4608], bias [4096]
- model.visual.deepstack_merger_list.{0,1,2}.norm.weight/bias [4608]
- model.visual.deepstack_merger_list.{0,1,2}.linear_fc1.weight [4608, 4608],
  bias [4608]
- model.visual.deepstack_merger_list.{0,1,2}.linear_fc2.weight [4096, 4608],
  bias [4096]
- 351 visual tensors + 24 merger tensors total, all present in manifest
- vision config: depth 27, hidden 1152, heads 16, head_dim 72, intermediate
  4304, out_hidden 4096, spatial_merge_size 2, patch_size 16, temporal_patch 2,
  num_position_embeddings 2304, deepstack_visual_indexes [8, 16, 24]
- text config: hidden 4096, layers 36, heads 32, kv_heads 36, head_dim 128,
  intermediate 12288, mrope_section [24,20,20], rope_theta 5000000

## Missing native components

- [x] native structs/API per visual conditioning (stage A) — src/model/vision.h
- [x] bind/resolve vision weights (stage B) — hd_vision_resolve in vision.c
- [ ] reference-image preprocessing tensors (pixel_values + grid) (stage C)
- [x] vision patch embed (stage D) — hd_vision_patch
- [x] vision transformer (27 blocks) (stage E) — hd_vision_forward block loop
- [x] vision merger -> image_embeds/deepstack (stage F) — merger + deepstack
- [ ] replace image placeholders in text embedding (stage G)
- [ ] integrate with existing ref_patches vinputs (stage H)
- [ ] oracle parity stage-by-stage (stage I)
- [ ] 1-step ref generation (stage J)
- [ ] full generation (stage K)

## Implementation status

- src/model/vision.h: hd_vision_binding, hd_vision_workspace, hd_vision_resolve,
  hd_vision_workspace_bytes, hd_vision_forward API.
- src/model/vision.c: resolve (all 351+24 vision weights), workspace layout,
  host orchestration of the full tower (patch -> pos -> rot -> 27 blocks ->
  merger + deepstack). GEMMs via hd_linear (cuBLAS production). Attention via
  cuDNN SDPA plan (ws->sdpa) with eager fallback.
- src/cuda/vision_kernels.cu/.h: device kernels (patch, layernorm, gelu,
  pos_interp, rot, rot_cos_sin, spatial_merge, qkv_split, attn_merge,
  masked_scatter, deepstack_inject).
- rot_pos_emb verified: output [n, 36] = cat([row_freq(18), col_freq(18)]),
  then emb = cat([rot, rot]) -> [n, 72], cos/sin fp32 [n, 72].
- pos_embed interp: 4-corner bilinear over 48x48 grid, then spatial-merge
  permute (the permute is folded into the row ordering of the interp output
  via build_pos_interp + the merge kernel).
- spatial merge: 2x2 unshuffle [n, 1152] -> [m, 4608], slot order
  (dh*m+dw)*hidden + c.
- deepstack mergers at blocks 8/16/24 (post-shuffle norm over 4608); final
  merger norm over 1152 BEFORE the merge view.
- TODO stage G: masked_scatter of image_embeds into text stream placeholder
  rows; deepstack injection into decoder layers 0/1/2.
- TODO stage C: produce pixel_values [n, 1536] bf16 from the resized ref
  (processor parity: smart_resize + rescale 1/255 + normalize mean/std 0.5 +
  patchify).

## Native primitives available (src/cuda/hd_cuda.h)

- hd_linear (bf16 GEMM, fp32 accum), hd_rmsnorm, hd_silu, hd_swiglu,
  hd_residual_add, hd_timestep_embed, hd_mrope_cos_sin, hd_apply_rotary,
  hd_attention_eager, hd_head_split/merge, hd_gather_rows,
  hd_apply_tms_condition, hd_scale_f32, hd_f32_convert_bf16,
  hd_bf16_buf_to_f32, hd_sched_* kernels
- MISSING: LayerNorm (vision uses it, not RMSNorm), GELU (gelu_pytorch_tanh),
  spatial merge (2x2 unshuffle), pos_embed interpolate, rot_pos_emb,
  masked_scatter, deepstack injection into decoder layers

## OPEN QUESTIONS

- [ ] exact rot_pos_emb coords (row,col) generation for grid [1,26,20] - see
  subagent A
- [ ] exact pos_embed_interpolate + spatial-merge permute order - see subagent A
- [ ] whether native pixel_values must match processor output EXACTLY (bf16
  rounding) - see subagent B
- [ ] vision tower runs in fp32 or bf16? (oracle captured fp32 hooks show
  float32; model runs bf16 autocast in pipeline) - see subagent B
- [ ] deepstack injection exact semantics at decoder layers 8/16/24 - see
  subagent A
- [ ] map upstream vision weights -> current o1 weight store - see subagent C