# M1 Execution Architecture

Single semantic execution path for the native HiDream forward: one production
implementation, device-resident ownership, direct tensor binding, persistent
reusable workspace, explicit scheduler boundary. Governs M1.3b (workspace
refactor) and M1.4. Matches `docs/M1_3A_PRE_FORWARD_ARCHITECTURE_FREEZE.md` §69.

## 1. Single semantic execution path

One production forward exists: `o1_model_forward()` composing the reference
CUDA primitives via the decoder-block chain, with the scheduler kept
separate as `o1_scheduler_step()`.

**Forbidden end-state:** duplicate implementations
(`o1_block_backend/device/reference/fast` each reimplementing semantics).
Unit tests may call primitives directly; production semantics exist **once**.
A diagnostic mode observes/dumps the *same* path via optional hooks —
never a duplicate math path.

## 2. Model init

`o1_model_init(profile, config_dir)`:

1. parse profile/config (`config/dev.json` or `base.json`)
2. load + validate weights (M1.1 `hd_weights_inventory` + `hd_weights_to_device`)
3. **bind tensors once** to stable typed descriptors (`o1_layer_weights[]`)
4. create CUDA stream(s) and cuBLAS/cuBLASLt handle(s) **once**
5. allocate persistent workspace **once** (profile-fixed shapes)

Init and workspace allocation happen at load; the forward hot path performs
**zero** allocations (see §13).

## 3. Weight binding

After load, each layer resolves its tensors to direct offsets into the
owned workspace/weight region. Conceptually:

```c
typedef struct {
    o1_tensor_ref input_norm;      /* RMSNorm weight [H]        */
    o1_tensor_ref q_proj, k_proj, v_proj, o_proj;  /* [.,H]     */
    o1_tensor_ref q_norm, k_norm;  /* RMSNorm [D]               */
    o1_tensor_ref gate_proj, up_proj, down_proj;   /* [FF,H]   */
} o1_layer_weights;
```

Binding **fails closed** on: missing tensor, dtype mismatch, shape mismatch,
duplicate binding, out-of-range offset.

**No string/JSON/hash lookup** and no `strcmp` scan inside any layer forward or
block path. All names are resolved once during init.

## 4. Workspace init

A model-specific persistent workspace is allocated once at load, sized by the
selected profile (see `docs/M1_MEMORY_LIFETIMES.md` §2/§8):

- ping-pong hidden pair (`hidden_A`, `hidden_B`)
- reusable scratch Q/K/V region
- persistent attention workspace (scores/probs/attn_out)
- persistent MLP workspace (gate/up/swi/down)
- final-norm + output-head buffers
- small scalar / device control buffers

No general-purpose allocator framework is built. The workspace is sufficient
to run a full transformer forward without runtime CUDA allocation.

## 5. Full forward device island

```
o1_model_forward(model, workspace, input, output, diag):
    # small host control metadata only
    upload request metadata (token ids, pos, timestep)      # H2D once
    embedding / target projection                           # device
    block 0  -> hidden_B                                    # device
    block 1  -> hidden_A                                    # device
    ...
    block 35 -> hidden_{A|B}                                # device
    final norm                                              # device
    output head -> x_pred                                   # device
    raw output stays on device until caller consumes it
```

Host responsibility is limited to: profile/config, tokenizer (M1.6),
launch orchestration, error handling, small scalar setup, diagnostics.
**No** D2H/H2D hidden-state, CPU residual/norm/MLP glue, or host tensor math
between blocks in normal forward.

## 6. Diagnostic hooks

Diagnostics observe the production path without duplicating it:

```c
o1_model_forward(..., diag = NULL);            /* production */
o1_model_forward(..., diag = &hooks);          /* diagnostic */
```

Hooks (active only when enabled): after_embedding, after_block_0,
after_block_mid, after_block_last, after_final_norm, after_output_head.
Hooks may copy selected tensors / compute metrics / record timing only in the
diagnostic build. The production build has **no** diagnostic overhead; a
separate diagnostic build enables CUDA events, tensor dumps, hashes (M2 uses
this for profiling).

## 7. Scheduler boundary

```
for step in timesteps:
    x_pred = o1_model_forward(state, timestep, conditioning, ...)  # device
    state  = o1_scheduler_step(state, x_pred, timestep)            # M1.5
```

Scheduler is a separate, independently testable module. Scheduler
coefficients/update rules are **not** mixed into decoder-block CUDA code.
M1.5 wires the denoise loop; M2 may move the scheduler onto device.

## 8. Error propagation

All new init/binding/workspace operations fail with actionable messages
(`hd_set_error`), including where available:

- profile, layer, tensor name
- expected shape vs actual shape
- requested bytes, buffer category
- CUDA/cuBLAS error string

A binding/allocation failure is reported at init — never deferred as silent
undefined behavior into the forward.

## 9. Stream/handle ownership

- **One** compute stream for the whole model lifetime (deterministic
  execution graph). No per-operation stream creation.
- **Persistent** cuBLAS/cuBLASLt handle(s) for the model lifetime (or another
  explicitly bounded lifetime); no creation/destruction per projection.
- Device constants / reusable descriptors created once.

Synchronization is explicit and minimal (see `docs/M1_FORWARD_CONTRACT.md` §17).

## 10. Resource cleanup

`o1_model_free()` deterministically releases, in order: device workspace,
weight buffers, cuBLAS handles, stream, driver state. Tests cover:

```
model init -> destroy
model init -> workspace init -> destroy
model init -> workspace init -> block fixture -> destroy
repeated block fixture -> destroy
```

No full model forward required to prove deterministic cleanup.

## 11. Persistent-process readiness

The model object supports:

```
load once
request 1 forward
reset request state
request 2 forward
...
destroy
```

Weights are loaded once and reused; no reload per forward. Request state
(prompt/tokens, image state, timestep, temp metadata) is separate from model
state. No server is built in M1.

## 12. CUDA Graph compatibility constraints

Keep the forward capture-friendly (no CUDA Graphs implemented yet):

- stable buffer addresses and layer-weight pointers after init
- profile-fixed shapes (no control flow depending on host tensor contents when avoidable)
- persistent streams/handles
- explicit request state
- **no** dynamic `cudaMalloc` inside forward, no per-layer pointer ownership
  churn, no host callbacks inside the block chain, no ad-hoc temp streams

CUDA Graph capture is an M2 task.

## 13. Forbidden hot-path operations

In the transformer block / denoise hot path, **zero** occurrences of:

| Pattern | Expected |
|---------|----------|
| `cudaMalloc` / `cudaFree` inside block or per-layer loop | 0 |
| `malloc` / `free` inside layer loop | 0 |
| `cudaDeviceSynchronize` after each primitive/block | 0 |
| D2H hidden state between blocks | 0 |
| H2D hidden state between blocks | 0 |
| string / `strcmp` / JSON / hash tensor lookup inside block loop | 0 |
| file read inside forward | 0 |
| weight transpose / repack inside forward (once at load) | 0 |
| cuBLAS/cuBLASLt handle creation inside layer loop | 0 |
| CUDA stream creation inside layer loop | 0 |

M1.3b audits and removes these from the current `block.c` path (in particular
the linear `strcmp` scan in `weight_ptr`), then revalidates the block fixture.

## 14. Explicit non-goals (M2/M3 backlog)

Deferred explicitly (see `docs/M2_CANDIDATES.md`, `docs/M3_CANDIDATES.md`):
CUDA Graphs, FlashAttention/custom attention, cuBLASLt tuning, kernel fusion,
shape specialization, bandwidth optimization, FP8/FP4 quantization, quality
suite, pinned/preload loader, large device arenas, soft/hard memory cap,
io_uring/preadv, GDS/cuFile.