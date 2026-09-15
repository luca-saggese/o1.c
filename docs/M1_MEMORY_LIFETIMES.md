# M1 Memory Lifetimes

Buffer inventory, lifetimes, aliasing and workspace sizing for the native
HiDream transformer forward. Complementary to `docs/M1_FORWARD_CONTRACT.md`;
governs M1.3b (device-resident workspace) and M1.4. Matches
`docs/M1_3A_PRE_FORWARD_ARCHITECTURE_FREEZE.md` §68.

## 1. Buffer inventory

| Buffer | Role | Shape | dtype |
|--------|------|-------|-------|
| `weights` | frozen model tensors | per-tensor | bf16 (cast at upload) |
| `input_ids` | token ids [bs,text_len] | int32 | host→device once |
| `position_ids` | MRoPE pos [3,bs,S] | int64→fp32 | device |
| `token_types` | gen mask [bs,S] | int64→fp32/bf16 | device |
| `attn_mask_4d` | [1,1,S,S] | bf16 | device (built per forward) |
| `input_embeds` | text embeddings [bs,T,H] | bf16 | device |
| `t_emb` | timestep embed [1,H] | bf16 | device |
| `vinputs` | image/noise [bs,img_tokens,C*P²] | bf16 | device |
| `vinputs_embedded` | x_embedder output [bs,img_tokens,H] | bf16 | device |
| `embeds_full` | cat(text,vinputs) [bs,S,H] | bf16 | device |
| `hidden_A` | ping-pong input buffer [S,H] | bf16 | device |
| `hidden_B` | ping-pong output buffer [S,H] | bf16 | device |
| `ln0` | input RMSNorm out [S,H] | bf16 | device (block scratch) |
| `qp,kp,vp` | Q/K/V projections pre-split | bf16 | device (block scratch) |
| `q,k,v` | head-split [Q|KV,S,D] | bf16 | device (block scratch) |
| `qr,kr` | post-RoPE Q/K | bf16 | device (block scratch) |
| `cos,sin` | MRoPE tables [1,S,D] | fp32 | device (block scratch) |
| `scores,probs` | attention [Q,S,S] | bf16 (+fp32 softmax) | device (block scratch) |
| `attn_sm` | attention out seq-major [S,Q,D] | bf16 | device (block scratch) |
| `attn_o` | o_proj out [S,H] | bf16 | device (block scratch) |
| `post` | post-attn norm [S,H] | bf16 | device (block scratch) |
| `gate,up,swi` | MLP [S,FF] | bf16 | device (block scratch) |
| `mlp_out` | down_proj out [S,H] | bf16 | device (block scratch) |
| `final_norm` | [S,H] | bf16 | device |
| `x_pred` | output head [S,3072] | bf16 | device (until scheduler) |

## 2. Size formula for each buffer

Sizes in **bytes**, bf16 = 2 bytes/elem unless marked fp32 (4).

```
hidden_S_H      = S * H * 2                          // 23*4096*2 = 188 KB (fast)
q_scalar        = Q * S * D * 2 = 32*S*128*2 = 8192*S
kv_scalar       = KV*S*D*2     = 8*S*128*2  = 2048*S
scores_probs    = Q * S * S * 2 = 32*S²*2 = 64*S²    // bf16; fp32 softmax temp
attn_sm         = S * Q * D * 2 = 8192*S
mlp_scalar      = S * FF * 2 = S*12288*2 = 24576*S
cos_sin         = 2 * (1*S*D*4) = 8*S*128          // fp32
attn_mask_4d    = S * S * 2                          // bf16 [1,1,S,S]
final_out       = S * 3072 * 2 = 6144*S
```

## 3. Lifetime class

| Class | Example |
|-------|---------|
| MODEL_LIFETIME | weights, bound layer refs, CUDA/cuBLAS handles, persistent workspace arena |
| REQUEST_LIFETIME | input_ids, position_ids, token_types, request state |
| FORWARD_LIFETIME | attn_mask_4d, input_embeds, hidden_A/B, final_norm, x_pred |
| BLOCK_LIFETIME | ln0, qp/kp/vp, q/k/v, qr/kr, cos/sin, scores/probs, attn_sm, gate/up/swi |
| OP_LIFETIME | loop-internal kernel temporaries (fp32 softmax overhead) |

Buffers are not held past last use merely for convenience.

## 4. Birth / last use

| Buffer | Birth | Last use |
|--------|-------|----------|
| weights | model init | whole forward |
| position_ids | request in | every block (RoPE) |
| token_types | request in | mask build (once/forward) |
| attn_mask_4d | forward start | all blocks (attention) |
| input_embeds | forward start | block 0 |
| t_emb | forward start | embedding merge (once) |
| vinputs_embedded | forward start | block 0 |
| embeds_full | embedding merge | block 0 |
| hidden_A/B | block 0 | final norm |
| q/k/v, qr/kr | per block | same block |
| cos/sin | per block | same block RoPE |
| scores/probs | per block | same block softmax |
| post/gate/up/swi | per block | same block MLP |
| final_norm | after block L-1 | head |
| x_pred | head | scheduler |

## 5. Read/write ownership

All transformer buffers read/write **on device**. Host only:
- writes request metadata (token ids, pos, timestep cache) before forward
- reads `x_pred` after forward if the caller needs host output (M1.5 scheduler)

No host read/write of hidden/intermediate transformer tensors in normal forward.

## 6. Alias candidates

Aliasing allowed only where lifetimes are provably disjoint and overwrite is safe:

| Alias group | Reasoning |
|-------------|-----------|
| `qp→q→qr` (in-place rotate) | rotate_half is elementwise per position; safe if q no longer needed |
| `kp→k→kr` (in-place rotate) | same |
| `attn_sm → attn_o` | attention output consumed by o_proj; original not needed |
| `post→hidden_out` | after MLP residual read, post is dead |
| `gate/up → swi` | swi = silu(gate)*up can reuse a dead operand |

Aliasing is **not** applied opportunistically in M1.3a; correctness first. These are candidates only, proven safe before enabling.

## 7. Ping-pong plan

```
block 0: hidden_A -> hidden_B
block 1: hidden_B -> hidden_A
block 2: hidden_A -> hidden_B
...
block 35 (L-1): hidden_{A|B} -> final_norm
```

- One hidden buffer pair is allocated at workspace init (FORWARD_LIFETIME).
- Residual source reads the input buffer; block output writes the other buffer.
- No full hidden-state output per layer is allocated.
- In-place block writes are **not** used (residual must survive; documented).

## 8. Profile-specific maximum sizes

Per-buffer dominant terms at each profile (bf16):

| Profile | S | hidden MB | scores+probs MB | mlp MB (per 2 ops) | attn_mask MB |
|---------|---|-----------|-----------------|--------------------|--------------|
| DEV-FAST (64) | 23 | 0.19 | 0.0005 | 0.56 | 0.001 |
| DEV-1024 | 1043 | 8.5 | 139.2 | 51.3 | 2.18 |
| DEV-2048 | 4115 | 33.7 | 2165.9 | 202.3 | 33.9 |

## 9. Simultaneous-live peak

Defined as the max sum of simultaneously-live temporary bytes, **not** the sum
of every temporary ever used. For one block the live set is:

```
peak_block = |ln0| + |qp,kp,vp| + |q,k,v| + |qr,kr| + |cos,sin(fp32)|
             + |scores| + |probs| + |attn_sm| + |attn_o| + |post|
             + |gate| + |up| + |swi| + |mlp_out| + ping-pong hidden pair
```

Dominant terms scale as `O(S²)` (scores+probs) + `O(S·FF)` (MLP) + `O(S·HQD)`.

## 10. 1024 theoretical peak

Using §9 with S=1043, Q=32, KV=8, D=128, H=4096, FF=12288:

```
scores        = 32 * 1043² * 2  ≈ 69.6 MB
probs         = 32 * 1043² * 2  ≈ 69.6 MB
mlp (gate+up)  = 2 * 1043*12288*2 ≈ 51.3 MB
hidden pair    = 2 * 1043*4096*2  ≈ 17.1 MB
q/k/v/qr/kr    ≈ (32+8+8) * 1043*128*2 ≈ 12.8 MB
attn_sm        = 1043*32*128*2 ≈ 8.5 MB
cos/sin (fp32) ≈ 1043*128*4*2  ≈ 1.1 MB
attn_mask      ≈ 2.18 MB
misc (ln0,attn_o,post,mlp_out,embeds) ≈ 40 MB
------------------------------------------------
peak ≈ 300-350 MB  (dominated by materialized attention O(S²))
```

## 11. 2048 theoretical peak

S=4115:

```
scores            = 32 * 4115² * 2   ≈ 1.083 GB
probs             = 32 * 4115² * 2   ≈ 1.083 GB
mlp (gate+up)     = 2 * 4115*12288*2 ≈ 202 MB
hidden pair       = 2 * 4115*4096*2  ≈ 67 MB
q/k/v/qr/kr       ≈ 48 * 4115*128*2  ≈ 50 MB
attn_sm           = 4115*32*128*2    ≈ 33.7 MB
attn_mask         ≈ 33.9 MB
misc              ≈ 180 MB
------------------------------------------------
peak ≈ 2.7-2.9 GB  (O(S²) attention dominates; scores+probs alone > 2 GB)
```

## 12. Reference attention O(S²) warning

The reference/materialized attention path (scores + probs both `[Q,S,S]`)
scales as `O(S²)`:

- DEV-1024: ~140 MB for scores+probs (acceptable on GB10)
- DEV-2048: ~2.17 GB for scores+probs alone (exceeds a comfortable per-forward
  budget and grows to O(S²) worst case)

**Decision:** M1.4 fast validation uses DEV-FAST (S=23). Production 2048 must
replace the materialized attention path (FlashAttention/custom attention /
chunked softmax) in **M2** before full-size runs. M1.3a records this, does not
silently allocate to OOM, and does not special-case a 2048 path now.

---

Per-profile workspace bytes and persistent buffer count are summarized in
`docs/M1_STATUS.md` at M1.3a closure.