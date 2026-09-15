# M1 Forward Contract

Frozen native forward call graph and topology for the HiDream O1 Image
engine, captured statically from the frozen Python oracle. Governs M1.4
(whole transformer forward) and beyond. This is a **contract**, not a
performance document — it fixes the semantic execution path that production
will keep, matching `docs/M1_3A_PRE_FORWARD_ARCHITECTURE_FREEZE.md` §67.

## 1. Oracle revision

- Upstream repo: `HiDream-ai/HiDream-O1-Image`
- Oracle checkout: `python/` (read-only, git-ignored)
- Oracle commit SHA: `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`
- Oracle working tree: clean
- Source of truth for call graph: `python/models/qwen3_vl_transformers.py`
  (`_forward_generation`, `Qwen3VLForConditionalGeneration`) and
  `python/models/pipeline.py` (`generate_image`, `build_t2i_text_sample`,
  `forward_once`).

## 2. Dev/Base model revisions

| Profile | HF repo | Immutable revision | Status |
|---------|---------|--------------------|--------|
| dev     | `HiDream-ai/HiDream-O1-Image-Dev-2604` | `b6acc2fe452b3120430620dc4354fa442ee081ea` | downloaded |
| base    | `HiDream-ai/HiDream-O1-Image` | `0b0901d99f200389e138c61946af1185f5f49a13` | `not_downloaded` (profile valid) |

Both share one transformer implementation (Dev/Base symmetry, `config/dev.json`/`config/base.json`).

## 3. Transformer topology (Dev)

- `hidden_size (H)` = 4096, `intermediate_size (FF)` = 12288
- `num_hidden_layers (L)` = 36
- `num_attention_heads (Q)` = 32, `num_key_value_heads (KV)` = 8
- `head_dim (D)` = 128, GQA groups = Q/KV = 4
- `vocab_size` = 151936, `max_position_embeddings` = 262144
- `rms_norm_eps` = 1e-6, `hidden_act` = SiLU (SwiGLU MLP)
- `rope_theta` = default, `rope_scaling` = `{mrope_interleaved: True, mrope_section: [24,20,20], rope_type: default}`
- Target: `patch_size (P)` = 32, `in_channels` = 3
- `bottleneck_dim` = H/4 = 1024
- `tms_token_id` = 151673
- Special tokens: image 151655, video 151656, vision_start 151652,
  `<|im_start|>` 151644, `<|im_end|>` 151645

## 4. Block topology

One decoder block = `Qwen3VLTextDecoderLayer`:

```
residual = x
ln0    = input_layernorm(x)                    RMSNorm(1e-6), [S,H]
q      = q_norm(q_proj(ln0)) -> head_split      q_proj: [H,H], q_norm RMSNorm
k      = k_norm(k_proj(ln0)) -> head_split      k_proj: [KV*D,H], k_norm RMSNorm
v      = v_proj(ln0) -> head_split              v_proj: [KV*D,H]
(q,k)  = apply_rotary(q, k, mrope cos/sin)     MRoPE on [Q/ KV, S, D]
attn_s = eager_attention(q,k,v,mask,scaling)    scaling = D^-0.5; FP32 softmax
attn   = o_proj(head_merge(attn_s))             o_proj: [H,H]; out [S,H] seq-major
h      = residual + attn                        residual add (bf16)
post   = post_attention_layernorm(h)            RMSNorm
x      = h + down(silu(gate(ln2)) * up(ln2))   SwiGLU MLP; gate/up [H,FF], down [FF,H]
```

Notes from oracle source:
- `q_norm`/`k_norm` applied **before** RoPE (confirmed `_forward_generation`
  + M1.3 fixture).
- `hd_attention_eager` output is already seq-major `[S,Q,D]`; **no**
  head_merge on the attention output (M1.3 fix).
- All projections bias-free in this model (verified in M1.3 weight load).

## 5. Input contract

`o1_model_forward(model, workspace, input, output, diag)` with:

- `model` — profile, bound layer weights, CUDA/cuBLAS resources, workspace (model lifetime)
- `input` — pre-tokenized canonical `input_ids` (int32, `text_seq_len`),
  `position_ids` [3, bs, seq], `token_types` [bs, seq], `timestep` (f32),
  `vinputs` (image/noise/target state) [bs, img_tokens, C*P²]
- frozen/explicit sequence metadata; tokenizer is NOT required by the
  forward (M1.6 tokenizer feeds it).
- `output` — device-resident `x_pred` [bs, seq, out_dim] + hooks.

## 6. Output contract

- Decoder output `hidden_states` [bs, seq, H] (bf16, device).
- `x_pred = final_layer2(hidden_states)` = `Linear(H, P*P*C)` =
  `Linear(4096, 3072)` → [bs, seq, 3072] (3·32·32). **_Not_ [bs,seq,H]**:
  FinalLayer output width is `P*P*out_channels` = 3072 (M1.3a documented
  deviation; oracle truth wins).
- M1.4 compares raw model output tensor, not final image bytes.
- `x_pred` is the model prediction consumed by the scheduler.

## 7. Sequence-length formulas (T2I)

Frozen pipeline (`build_t2i_text_sample`), batch 1:

- `image_len = (H_img // P) * (W_img // P)` = `(H//32)*(W//32)` target image tokens
- template = `apply_chat_template(user)[…] + boi_token + tms_token*TIMESTEP_TOKEN_NUM`
  with `TIMESTEP_TOKEN_NUM = 1`
- `text_len = len(tokenizer(template, add_special_tokens=False))`
  (template already includes the trailing boi + 1×tms)
- `seq_len (S) = text_len + image_len`
- canonical fixture: `text_len = 19` (17 prompt ids + boi + tms), `image_len = 4`
  → `S = 23`
- `token_types[0, text_len-1 : text_len-1+image_len+TIMESTEP_TOKEN_NUM] = 1`
  (gen region incl. the tms position), rest 0 (AR text); `vinput_mask = (token_types==1)`
- `position_ids` [3, 1, S] from `get_rope_index_fix_point` (interleaved MRoPE,
  sections [24,20,20]) — text, then image tokens with per-axis grid positions.

## 8. 1024 shape profile (DEV-1024)

- resolution 1024×1024 (assumed square for the reference profile; aspect via
  `find_closest_resolution` if configured)
- `h_patches = w_patches = 1024//32 = 32`
- `image_len = 32*32 = 1024`
- `text_len = 19` (canonical prompt)
- `S = 19 + 1024 = 1043`
- attention scores O(Q·S²): 32 · 1043² · 2 bytes ≈ 69.6 MB (scores) + probs ≈ 69.6 MB
- peak workspace: dominated by S² attention materialization (~140 MB per large
  array family); see `docs/M1_MEMORY_LIFETIMES.md`.

## 9. 2048 shape profile (DEV-2048)

- resolution 2048×2048 (square reference)
- `h_patches = w_patches = 2048//32 = 64`
- `image_len = 64*64 = 4096`
- `text_len = 19` (canonical prompt)
- `S = 19 + 4096 = 4115`
- attention scores O(Q·S²): 32 · 4115² · 2 bytes ≈ 1.08 GB (scores) + probs ≈ 1.08 GB
- **Reference attention O(S²) scales to >2 GB just for scores+probs at 2048.**
  M1.4 fast validation runs at DEV-FAST (64); production 2048 must replace the
  materialized attention path in M2 before full-size runs. Recorded; not a
  blocking M1.3a decision (no 2048 run in M1.3a).

## 10. GQA contract

- `Q = 32`, `KV = 8`, `D = 128`, groups per KV = `Q/KV = 4`
- stored K/V after projection: [KV, S, D] before repeat_kv
- logical attention: repeat_kv to [Q, S, D]; scores/probs [Q, S, S]
- physical K/V duplication only if the reference path requires it; M2 may
  reintroduce grouped indexing. Memory impact of the reference expansion is
  documented in `docs/M1_MEMORY_LIFETIMES.md`.

## 11. MRoPE contract

- `inv_freq` from default rope init (theta 10000), dtype fp32
- `apply_interleaved_mrope(freqs, [24,20,20])`: per-axis 3×section values,
  interleaved merge T/H/W → seq axis
- `freqs = inv_freq @ pos_ids` (fp32); `emb = cat((freqs,freqs),-1)`
- `cos = emb.cos() * attention_scaling`, `sin = emb.sin() * attention_scaling`
- shape `[bs, seq, head_dim] = [1, S, 128]`; cast to input dtype (bf16) at use
- rotation applied on head-major `[Q,S,D]` (q) / `[KV,S,D]` (k) **after** q/k
  norm and **before** softmax attention (verified vs M1.2/M1.3 fixtures).

## 12. Residual contract

- residual source = block input `x` (kept alive across the block)
- attention residual added on h: `h = residual + o_proj(attn)`
- MLP residual added on h: `x_out = h + down(silu(gate*h)*up(h))`
- with hidden_A/hidden_B ping-pong, residual must read from the **input**
  buffer and write to the **other** buffer each block; no in-place overwrite
  of a value still needed (see `docs/M1_MEMORY_LIFETIMES.md` §7).

## 13. MLP contract

- gate = `gate_proj(post_ln)` [S, FF]; up = `up_proj(post_ln)` [S, FF]
- gated = `silu(gate) * up` (SwiGLU) [S, FF]
- out = `down_proj(gated)` [S, H]
- residual add; no fusion in M1.

## 14. Output-head contract

- final norm: `norm(hidden_states)` RMSNorm → [S, H]
- head: `final_layer2.linear` Linear(4096 → 3072) → `x_pred` [S, 3072]
- `x_pred` stays on device until scheduler consumes it; dtype bf16 (compute),
  fp32 only on host image encoding (M1.6+ / M1.7).
- no target-token slicing in the transformer; slicing/masking happens via
  `token_types`/`vinput_mask` in the caller.

## 15. Scheduler boundary

```
for step in timesteps:
    x_pred = model_forward(state, model_timestep, conditioning)   # device
    state  = scheduler_step(state, x_pred, scheduler_timestep)    # device or host (M1.5)
```

- Scheduler is semantically separate; coefficients/step rules are **not** mixed
  into decoder-block CUDA code (`docs/M1_3A… §41`).
- M1.5 implements `o1_scheduler_step()`; flash scheduler (28 steps, timesteps
  frozen in `config/startup_manifest_dev.json`).

### 15.1 Timestep domain chain (binding, frozen oracle semantics)

`o1_model_forward()` accepts the **model timestep** domain (PixelDiT time),
NOT scheduler time. The unit conversion lives in the scheduler/control layer:

```
scheduler domain:    step_t        = 999            (manifest scheduler_timestep)
        ↓
sigma domain:        sigma         = 999/1000 = 0.999
                                     (clamped to ≥ T_EPS = 0.001)
        ↓
model domain:        model_timestep = 1 - sigma ≈ 0.001
                                     (what o1_model_forward receives)
        ↓
embedder input:      t_scaled      = model_timestep * 1000 ≈ 1.0
                                     (hd_forward multiplies ×1000 internally;
                                      TimestepEmbedder.forward does t*1000)
```

Steps: `step_t 999 → model_t ≈ 0.001 → embedder ≈ 1.0`; `987 → 0.013 → 13`;
`974 → 0.026 → 26`. **Never** pass scheduler time (999) straight into
`o1_model_forward`: that produced embedder input 999000 in M1.4 (wrong by ~6
orders) and is the accepted root cause of the M1.4 timestep_conditioning
divergence.

`scheduler_step_t`, `sigma`, `model_timestep`, `timestep_embedder_input` are
all persisted per captured step (M1.5 manifests) to remove the `timestep: 999`
ambiguity.

### 15.2 Masked-row x_pred contract

`forward_once` returns **only the vinput-masked rows** of x_pred:
`x_pred[0, token_types>0]` → shape `[1, IMG, 3072]` (4 image rows, not the
full 23-row sequence). The scheduler's v_cond/guidance/`sched.step` arithmetic
operates exclusively on these image rows. Native must slice rows 19..22 from
the full `[23, 3072]` x_pred before any scheduler math, matching the
`vinput_mask` (token_types binary, image tokens set).

## 16. CPU/GPU ownership map

| Stage | Owner |
|-------|-------|
| profile/config/parse | host |
| tokenizer (prompt→ids) | host (M1.6) |
| input_embeds (text embed) | device |
| timestep conditioning (t_embedder1) | device |
| image/noise patch projection (x_embedder) | device |
| transformer blocks 0..35 | device |
| final norm + head | device |
| x_pred | device (until scheduler) |
| scheduler step | host in M1.5 (device later) |

## 17. Allowed synchronization points

- before forward, if input upload must complete
- between dependent kernels via stream ordering (single compute stream)
- after final output only when host consumes it
- diagnostic-only sync when instrumentation enabled

**Forbidden:** `cudaDeviceSynchronize` after each primitive or block.

## 18. M1.4 diagnostic checkpoint plan

One whole-model Python forward captures all of (in one run):

1. embedding output
2. block 0 output
3. block middle output (e.g. layer 17)
4. block last output (layer 35)
5. final norm input
6. final norm output
7. final output-head result (x_pred)

Debugging strategy: compare embedding → early block → middle → late → final;
on divergence, return to the cheapest V2/V3 reproduction at the first deviant
boundary. Do not rerun the Python full model for every native change.