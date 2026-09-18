# M1.3a — Pre-Forward Architecture Freeze

## Position in the roadmap

This sub-milestone is executed **after M1.3 is green and before M1.4 begins**.

Required entry state:

```text
M1.0 PASS
M1.1 PASS
M1.2 PASS
M1.3 PASS
```

M1.3a exists to prevent the first complete transformer forward from being built on an execution architecture that is known in advance to require major redesign during M2/M3.

The purpose is **not performance optimization**.

The purpose is to freeze:

- the real forward topology;
- execution ownership;
- tensor binding;
- real shapes;
- buffer lifetimes;
- reusable workspace;
- the device-resident execution contract;
- the boundary between transformer and scheduler;
- diagnostic hooks;
- constraints that preserve future CUDA Graph capture;
- the exact preconditions for M1.4.

The central rule is:

> Build M1.4 on the same semantic execution architecture that production will keep, even if its kernels are still unoptimized.

---

# 1. Source of truth

Before changing code, read:

```text
MILESTONES.md
docs/WORKING_MODE.md
docs/M1_STATUS.md
docs/M1_NUMERICAL_CONTRACT.md
docs/M1_REPORT.md            # if it already exists partially
config/oracle.lock
config/models.lock
```

Also inspect:

```text
/python
```

at the exact frozen oracle revision.

Do not modify `/python`.

Use the already-frozen M1.3 fixtures and manifests as correctness references.

---

# 2. Entry gate

Do not start M1.3a if M1.3 is not green.

Verify:

```text
M1.3 block fixture passes
critical block internals pass their numerical classes
native block is leak-free across repeated calls
oracle working tree clean
engine repository clean
```

Record in `docs/M1_STATUS.md`:

```text
M1.3 COMPLETE
M1.3a STARTED
```

with the current engine commit.

---

# 3. Run-budget rule

M1.3a must not increase the expensive Python run budget.

Allowed:

```text
static inspection
host-only tests
manifest parsing
shape calculations
buffer-layout tests
native M1.3 fixture reruns
native synthetic tests
CUDA allocation tests
compute-sanitizer on small native fixtures
```

Forbidden:

```text
new whole-model Python forward
new denoise sequence
new full image generation
new early/middle/late Python block captures
```

The first planned whole-model Python forward remains M1.4.

If M1.3a discovers that an existing M1.3 golden is insufficient to verify a structural refactor, first determine whether the property can be proven with existing tensors or synthetic native tests.

Do not regenerate Python fixtures merely because C/CUDA code changed.

---

# 4. Mandatory outputs

M1.3a must produce:

```text
docs/M1_FORWARD_CONTRACT.md
docs/M1_MEMORY_LIFETIMES.md
docs/M1_EXECUTION_ARCHITECTURE.md
```

and, if not already available:

```text
artifacts/m1/shape_inventory.json
artifacts/m1/buffer_plan.json
```

The `artifacts/` files may remain git-ignored.

The compact architectural documents must be versioned.

---

# 5. M1.3a deliverables at a glance

Before M1.4, the repository must have an explicit answer to all of these:

```text
What is the exact forward call graph?
What tensors enter the transformer?
What tensors leave it?
Which operations run on host?
Which operations run on device?
Which tensors remain device-resident?
Which tensor pointers are fixed after init?
Which buffers are persistent?
Which buffers can alias?
Which buffers ping-pong?
What are the real 1024 and 2048 shapes?
What is the actual sequence length at each supported profile?
What is the maximum live memory for one forward?
Where is the scheduler boundary?
Where are allocations permitted?
Where are copies permitted?
Can diagnostics observe the production path without duplicating it?
Does the architecture remain compatible with future CUDA Graph capture?
```

M1.4 must not begin until these are answered.

---

# 6. M1.3a.1 — Freeze the real upstream call graph

## Goal

Produce a complete static map of the frozen oracle's T2I execution path.

Do not infer the call graph from model names.

Trace the actual frozen Python source.

At minimum cover:

```text
prompt/token preparation
position / sequence metadata
input embeddings
target/image patch preparation
target/image projection
timestep / conditioning preparation
transformer block chain
final normalization
output/pixel projection
output reshape / reconstruction boundary
scheduler boundary
```

For each transformer block map:

```text
input normalization
Q projection
K projection
V projection
Q/K normalization if present
MRoPE / position transform
attention score path
attention mask
softmax
attention value aggregation
output projection
attention residual
FFN / gated MLP
MLP residual
block output
```

Do not assume all blocks are identical until static inspection proves it.

If layer-specific behavior exists, classify block types.

---

# 7. Call graph format

Create `docs/M1_FORWARD_CONTRACT.md`.

For every graph node record:

```text
node_id
source function/module
input tensors
output tensors
shape formula
dtype
layout
CPU or GPU ownership
called once / once per forward / once per denoise step
persistent state touched
temporary workspace required
synchronization requirement
```

Use formulas instead of only one concrete size when a dimension varies.

Example:

```text
hidden:
    shape = [B, S, H]
    dtype = BF16
    owner = GPU
    lifetime = embedding_output -> final_norm
```

---

# 8. No duplicate semantic path

M1.3a must establish one production execution architecture.

Forbidden end-state:

```text
o1_block_backend()
o1_block_device()
o1_block_reference()
o1_block_fast()
```

when each independently implements transformer semantics.

Preferred:

```text
o1_block_forward()
```

with optional instrumentation.

Unit tests may call primitives directly.

Production semantics must exist once.

A diagnostic mode may observe/dump/timestamp the same path.

---

# 9. Required execution API structure

The implementation should converge toward a semantic structure equivalent to:

```text
o1_model_forward()
    -> prepare already-bound model state
    -> embedding / target projection
    -> block chain
    -> final normalization
    -> output projection
```

and separately:

```text
o1_scheduler_step()
```

Later M1.5 may compose them into the denoising loop.

Do not mix scheduler arithmetic into transformer block implementation.

---

# 10. M1.3a.2 — Freeze real shape profiles

Create a real shape inventory before the first full forward.

At minimum define:

```text
DEV-FAST
DEV-1024
DEV-2048
BASE-FAST or BASE-compatible symbolic profile
```

`DEV-FAST` is the existing cheap validation resolution selected by the M1 plan.

If `DEV-FAST == 1024`, do not create a duplicate profile.

---

# 11. Resolve image-token ambiguity from the frozen oracle

The repository must explicitly distinguish:

```text
vision/reference-image preprocessing tokenization
```

from:

```text
target generative image-token / patch sequence
```

Do not use a patch size from the vision encoder to dimension the target generative transformer path unless the frozen T2I oracle actually does so.

Static inspection must determine and document:

```text
target patch dimensions
target image token formula
vision/reference token formula if different
merge behavior
text token contribution
special token contribution
timestep/conditioning contribution
total sequence-length formula
```

This is a required M1.3a gate because attention workspace sizing depends on it.

---

# 12. Shape inventory contents

For each supported profile record:

```text
resolution
batch
guidance mode
text token count for canonical fixture
image/target token count
total sequence length
hidden width
Q head count
KV head count
head dimension
MLP intermediate width
layer count
Q matrix shape
K matrix shape
V matrix shape
attention-score shape
attention-output shape
MLP gate/up shapes
MLP intermediate shape
block output shape
final output shape
```

For variable text length, record:

```text
canonical_fixture_length
supported_max_length
formula
```

Do not hardcode only the canonical prompt length into runtime topology.

---

# 13. Repetition counts

For each important operation record how often it occurs:

```text
per block
per forward
per denoise step
per full Dev generation
per full Base generation
```

Example concept:

```text
RMSNorm:
    X per block
    X * layers per forward
    X * layers * steps per generation
```

This is not yet an optimization exercise.

It exists to identify structural hot paths and prevent designs that introduce repeated allocations/copies into operations executed thousands of times.

---

# 14. M1.3a.3 — Freeze tensor binding

String-based safetensors names are acceptable during:

```text
manifest parsing
weight import
model binding
diagnostics
```

They are not acceptable as the normal runtime lookup mechanism inside the transformer loop.

After model initialization, bind all required tensor descriptors to stable typed structures.

Conceptually:

```c
typedef struct {
    o1_tensor_ref q_proj;
    o1_tensor_ref k_proj;
    o1_tensor_ref v_proj;
    o1_tensor_ref o_proj;

    o1_tensor_ref q_norm;
    o1_tensor_ref k_norm;

    o1_tensor_ref gate_proj;
    o1_tensor_ref up_proj;
    o1_tensor_ref down_proj;

    o1_tensor_ref input_norm;
    o1_tensor_ref post_attn_norm;
} o1_layer_weights;
```

Use the actual frozen topology, not this example blindly.

---

# 15. Tensor binding rules

After init/load:

```text
layer[i].attn.q_proj
layer[i].attn.k_proj
...
```

or equivalent offsets/pointers must resolve directly.

Do not do:

```text
hash lookup by tensor name
JSON lookup
string concatenation
unordered map lookup
```

inside each layer forward.

The binding stage must fail closed if:

```text
required tensor missing
unexpected incompatible dtype
shape mismatch
duplicate binding
invalid offset
```

---

# 16. Binding validation

Add host-only tests that verify:

```text
all required fields bound
all layer counts correct
all shapes match config
all dtypes match config
no pointer references outside owned model regions
Dev binding complete
Base binding structure valid
```

If Base weights are not yet resident, validate the mapping schema/profile structurally.

---

# 17. M1.3a.4 — Define the device island

The full transformer forward should be device-resident in terms of data ownership.

This does **not** require one giant CUDA kernel.

It is acceptable for the CPU to orchestrate launches:

```text
launch norm
launch GEMM
launch RoPE
launch attention
launch MLP
```

The mandatory property is:

```text
intermediate transformer tensors remain on device
```

---

# 18. Device-island contract

Normal forward:

```text
small host control metadata
        ↓
required initial H2D
        ↓
device-resident embedding / projection
        ↓
device-resident block 0
        ↓
device-resident block 1
        ↓
...
        ↓
device-resident block N
        ↓
device-resident final head
        ↓
raw output stays on device until caller explicitly needs host output
```

Forbidden between normal blocks:

```text
D2H hidden state
CPU residual math
CPU normalization
CPU MLP glue
H2D hidden state
```

---

# 19. What may remain host-side in M1

Acceptable host responsibilities:

```text
profile selection
validated config
tokenizer before model call
launch orchestration
error handling
small immutable scalar setup
diagnostic configuration
```

Scheduler remains semantically separate and may remain host-side until M1.5/M2 unless its algorithm already requires device-side data operations.

---

# 20. M1.3a.5 — Build the memory/lifetime diagram

Create `docs/M1_MEMORY_LIFETIMES.md`.

At minimum map:

```text
weights
token/prompt state
target/image tokens
timestep conditioning
hidden input
hidden A
hidden B
Q
K
V
post-RoPE Q
post-RoPE K
attention scores or reference attention workspace
attention output
MLP gate
MLP up
MLP intermediate
MLP output
residual state
final norm scratch
final output
scheduler state boundary
```

For every buffer record:

```text
birth
first use
last use
size formula
alignment requirement
read/write mode
can alias?
can overwrite input?
persistent?
per-forward?
per-request?
```

---

# 21. Lifetime classes

Use clear classes such as:

```text
MODEL_LIFETIME
REQUEST_LIFETIME
FORWARD_LIFETIME
BLOCK_LIFETIME
OP_LIFETIME
```

Do not keep an activation alive longer than its last required use merely because it is convenient.

Do not aggressively alias until lifetime safety is proven.

Correctness comes first.

---

# 22. M1.3a.6 — Define the minimal persistent workspace

Before M1.4, create a simple model-specific workspace.

Do **not** build a general-purpose allocator framework.

The workspace should be sufficient to execute a full transformer forward without runtime CUDA allocation.

Candidate categories:

```text
hidden_A
hidden_B
Q
K
V
attention scratch
attention output
MLP gate/up/intermediate scratch
final output
small scalar/device control buffers
```

Use actual lifetime analysis to reduce this list where safe.

---

# 23. Ping-pong hidden state

Prefer:

```text
hidden_A
hidden_B
```

and swap block-by-block.

Conceptually:

```text
block 0: A -> B
block 1: B -> A
block 2: A -> B
...
```

Do not allocate one full hidden-state output per layer.

If the current validated block implementation writes in-place safely, document why.

Otherwise keep explicit ping-pong.

---

# 24. Workspace sizing policy

Workspace sizing must be derived from the selected execution profile.

For each profile calculate:

```text
hidden bytes
Q bytes
K bytes
V bytes
attention score/scratch bytes
attention output bytes
MLP intermediate bytes
final output bytes
total simultaneously-live temporary bytes
```

Record:

```text
worst-case live bytes
not sum of every temporary ever used
```

This distinction matters.

---

# 25. 1024 and 2048 memory model

Before M1.4, compute theoretical workspace requirements for:

```text
1024
2048
```

using the frozen real shape formulas.

Do not wait for a 2048 run to discover that the reference attention implementation requires an unexpectedly large score matrix.

M1.3a does not require executing a 2048 full forward.

It requires proving the memory formula.

---

# 26. Reference attention workspace warning

M1 may still use a transparent/materialized attention implementation for correctness.

If attention-score materialization scales as:

```text
O(S^2)
```

record the actual bytes at each profile.

If 2048 would exceed practical memory or the project's memory policy, document that M1.4 fast validation uses the cheap profile and that M2 must replace the reference attention path before production 2048.

Do not silently allocate until OOM.

---

# 27. M1.3a.7 — Allocation policy

From M1.3a onward, define:

```text
cudaMalloc/cudaFree inside transformer block = architectural bug
cudaMalloc/cudaFree inside per-layer loop = architectural bug
malloc/free inside transformer hot path = architectural bug
```

Allocations belong in:

```text
model initialization
profile/workspace initialization
request preparation if unavoidable
```

not normal block execution.

---

# 28. Native allocation audit

Before M1.4, inspect every primitive/block used by the forward.

Produce a table:

```text
function
allocates host memory?
allocates device memory?
frees memory?
creates CUDA event?
creates stream?
creates cuBLAS handle?
```

Any repeated allocation/resource creation in the block path must be removed or justified before M1.4.

---

# 29. Handle/resource lifetime

Create once and reuse where applicable:

```text
CUDA stream(s)
cuBLAS/cuBLASLt handles
reusable descriptors if stable
device constants
workspace
```

Do not create/destroy these per layer.

This is architectural hygiene, not M2 tuning.

---

# 30. M1.3a.8 — Weight-layout policy

M1.3a must document the runtime weight layout contract.

For every major linear tensor family record:

```text
on-disk shape
logical mathematical shape
runtime layout
transpose flag
whether load-time conversion occurs
```

Rule:

```text
no transpose/repack/swizzle per denoise step
```

If the existing M1 loader uses the safetensors layout directly, document it.

If a conversion is needed for correctness or stable GEMM calling convention, perform it once at load/init.

Do not introduce a performance-oriented packed format yet unless already required by M1.

---

# 31. M1.3a.9 — Static audit of generic CUDA assumptions

Before the full forward multiplies every primitive across all layers, audit kernels for obviously unsuitable launch structure.

Check especially:

```text
batch = 1 paths
one-row work mapped to one useful thread
huge early-return ratios
serial reductions
shape-dependent branches
wrong head grouping assumptions
hardcoded Dev-only dimensions
```

This is a static/structural audit.

Do not start shape-specialized optimization in M1.3a.

If a kernel is functionally correct but clearly inefficient, record it in:

```text
docs/M2_CANDIDATES.md
```

or equivalent backlog.

Only fix it now if its structure would make M1.4 impractical or incorrect.

---

# 32. M1.3a.10 — Preserve CUDA Graph compatibility

Do not implement CUDA Graphs in M1.3a.

Do make the full-forward architecture capture-friendly.

Avoid unnecessary:

```text
dynamic cudaMalloc inside forward
changing device pointer ownership every layer
host callbacks inside block chain
ad-hoc temporary stream creation
random lifetime-dependent pointers
control flow that depends on host tensor contents when avoidable
```

Prefer:

```text
stable buffer addresses
stable layer-weight pointers
profile-fixed shapes
persistent streams/handles
explicit request state
```

CUDA Graph implementation remains M2.

---

# 33. M1.3a.11 — Separate production and diagnostics semantically, not mathematically

Diagnostics must not create a duplicate model implementation.

Preferred architecture:

```text
production:
    forward(..., diag = NULL)

diagnostic:
    forward(..., diag = &hooks)
```

Possible hooks:

```text
after_embedding
after_block_0
after_block_mid
after_block_last
after_final_norm
after_output_head
```

Hooks may:

```text
copy selected tensors
compute diagnostic metrics
record timing
```

only when diagnostics are enabled.

---

# 34. No diagnostic overhead in normal production build

Plan now for:

```text
production build
diagnostic build
```

using the same execution path.

It is acceptable for diagnostic builds to enable:

```text
extra CUDA events
tensor dumps
hashes
NVTX
```

but normal performance measurements must disable them.

M1.3a only establishes the structure.

M2 will use it for profiling.

---

# 35. M1.3a.12 — Fixture strategy before M1.4

Do not create three separate Python block runs now.

M1.3 already has a complete block fixture.

Use it to validate structural refactors.

The M1.4 single whole-model Python capture will later collect:

```text
embedding
early block output
middle block output
late block output
final norm
final head
complete output
```

in one expensive oracle execution.

This preserves the run budget.

---

# 36. Optional extra native block coverage

If static inspection proves that there are distinct block classes, M1.3a may create:

```text
native synthetic fixtures
```

or reuse weights/inputs already available to exercise different binding structures.

Do not run Python solely to generate early/middle/late block fixtures unless a true semantic block variant cannot otherwise be validated.

Record any such exception explicitly.

---

# 37. M1.3a.13 — Freeze the one-forward input contract

Before M1.4, define the native full-forward API contract.

It must accept explicit:

```text
model/profile
pre-tokenized canonical input
position/sequence metadata
timestep
image/noise/target state
output destination
diagnostic hooks optional
```

Do not require the native tokenizer yet.

Tokenizer remains M1.6.

This keeps M1.4 isolated from text-processing differences.

---

# 38. Suggested C-level contract

Exact naming may follow repository conventions.

Conceptually:

```c
typedef struct {
    const int32_t *token_ids;
    size_t token_count;

    const void *image_state;
    int width;
    int height;

    float timestep;

    /* frozen/explicit sequence metadata */
} o1_forward_input;

typedef struct {
    void *device_output;
    size_t output_bytes;
} o1_forward_output;

o1_status o1_model_forward(
    o1_model *model,
    o1_forward_workspace *workspace,
    const o1_forward_input *input,
    o1_forward_output *output,
    const o1_diag_hooks *diag);
```

Do not copy this signature blindly if the real frozen model needs different inputs.

The point is explicit ownership and no hidden global mutable state.

---

# 39. Request state versus model state

Separate:

```text
model state
```

from:

```text
request state
```

Model state includes:

```text
weights
config
bound layer descriptors
persistent handles
profile-independent metadata
```

Request state includes:

```text
prompt/tokens
current image/noise state
timestep
temporary sequence metadata
request workspace state
```

This separation prepares for persistent-process inference without implementing a server now.

---

# 40. M1.3a.14 — Persistent-process readiness

Do not build the server in M1.3a.

Do ensure that the model object can conceptually support:

```text
load once
request 1 forward
reset request state
request 2 forward
...
destroy
```

The full model must not require reloading weights for each forward.

A repeated native M1.3 block call should already satisfy this principle.

M1.4 should inherit it.

---

# 41. M1.3a.15 — Scheduler boundary contract

Document in `docs/M1_FORWARD_CONTRACT.md`:

```text
transformer input state
transformer output prediction
scheduler input
scheduler output updated state
```

Scheduler must remain separately testable.

Do not mix scheduler coefficients or update rules into decoder-block CUDA code.

The eventual denoise loop should conceptually be:

```text
for timestep:
    prediction = model_forward(state, timestep, conditioning)
    state = scheduler_step(state, prediction, timestep)
```

even if later optimization fuses some device operations.

---

# 42. M1.3a.16 — Benchmark protocol freeze, without running it

Define a future stable benchmark identity such as:

```text
O1_REFERENCE_0
```

Record:

```text
model: Dev
resolution: chosen reference resolution
steps: 28
seed: fixed
prompt: canonical
batch: 1
warm/cold model condition explicitly defined
precision: BF16 reference
```

Metrics to be used later:

```text
startup
prompt processing
one-forward latency
per-step median
per-step p95
total generation
peak memory
final image/quality artifact
```

Do **not** execute the 28-step benchmark in M1.3a.

The purpose is to stop benchmark protocol drift later.

---

# 43. M1.3a.17 — Bandwidth/accounting hooks, not optimization

For future M2 profiling, make it possible to attribute dominant operations.

For each major primitive, ensure its tensor descriptor knows enough to calculate:

```text
bytes read
bytes written
logical FLOPs where meaningful
shape
```

Do not add heavy per-kernel timers to production M1.3a.

A diagnostic build may expose hooks.

The actual GB/s analysis belongs to M2.

---

# 44. M1.3a.18 — Loader architecture: constrain, do not optimize yet

Do not implement the M3 pinned `pread` loader in M1.3a.

But do prevent M1.4 from depending on pathological loader behavior.

Required architectural properties:

```text
weights are fully bound before forward
forward never opens model files
forward never parses safetensors names
forward never performs weight-layout conversion
forward never loads a weight on demand from disk
```

Future loader optimization can then replace the load mechanism without changing transformer semantics.

---

# 45. Memory-cap policy: document future requirement, do not hardcode q38 values

Do not copy:

```text
soft = 114 GiB
hard = 118 GiB
```

into M1.3a.

Those values belonged to a different engine.

Instead record that M2/M3 must derive safe limits from:

```text
resident weights
persistent workspace
activation peak
pinned staging
CUDA/runtime overhead
system reserve
```

M1.3a must at least compute its own forward workspace requirement so later memory policy has reliable inputs.

---

# 46. Quantization policy

M1.3a remains BF16-first.

Do not introduce:

```text
FP8
FP4
NVFP4
weight-only quantization
activation quantization
```

before the BF16 full forward is correct.

Document the future rule:

```text
quantized modes must use the same semantic execution path
```

and require their own numerical/quality contract later.

---

# 47. Quality-suite policy

Do not generate a 10–20 prompt quality corpus now.

Record it as a prerequisite before aggressive quantization or quality-affecting precision work.

M1.3a correctness remains tensor-based.

---

# 48. M1.3a.19 — Exact forbidden patterns before M1.4

Search the codebase for patterns equivalent to:

```text
cudaMalloc inside block
cudaFree inside block
malloc inside layer loop
free inside layer loop
cudaDeviceSynchronize inside layer loop
D2H hidden state between blocks
H2D hidden state between blocks
tensor lookup by string inside block loop
file read inside forward
weight transpose inside forward
handle creation inside layer loop
stream creation inside layer loop
```

For each occurrence:

```text
remove
or document why it is not in the hot execution path
```

---

# 49. Synchronization policy

M1.3a must not introduce synchronization merely for convenience.

Allowed synchronization points should be explicit.

At minimum document:

```text
before forward, if input upload requires completion
between dependent kernels through stream ordering
after final output only when host actually consumes it
diagnostic-only sync when instrumentation requires it
```

Avoid:

```text
cudaDeviceSynchronize after each primitive
cudaDeviceSynchronize after each block
```

Correctness should rely on stream dependency ordering where possible.

---

# 50. Stream policy

Keep stream design simple in M1.

A single compute stream is acceptable.

The goal is not overlap optimization.

The goal is to avoid per-operation stream creation and to keep the execution graph deterministic.

If the existing code already uses more than one stream, document:

```text
stream purpose
ownership
dependency mechanism
```

---

# 51. cuBLAS/cuBLASLt policy

Handles must be persistent for the model/runtime lifetime or another clearly bounded lifetime.

Do not create/destroy a handle for each projection.

Document:

```text
compute type
input type
weight type
output type
transpose/layout convention
workspace ownership
```

for each major GEMM family.

This is necessary before M1.4 multiplies any layout mistake across all layers.

---

# 52. MRoPE / position contract

Before M1.4, freeze:

```text
position-id construction
MRoPE sections
dimension ordering
broadcast rules
dtype
where rotation occurs
Q/K normalization order relative to rotation
```

Use the frozen M1.2/M1.3 fixtures to ensure the implementation order remains correct.

No new Python forward is required.

---

# 53. GQA contract

Document:

```text
Q head count
KV head count
Q-per-KV grouping
head dimension
stored K/V shape
expanded/logical attention shape
```

Do not physically duplicate K/V merely to simplify indexing unless the existing reference path requires it for correctness and the memory impact is documented.

A later M2 may optimize representation.

---

# 54. Residual contract

Freeze exact residual semantics for one block:

```text
which tensor is residual source
when normalization happens
where attention output is added
where MLP output is added
whether any branch is in-place
```

This is especially important when introducing hidden_A/hidden_B ping-pong.

Do not introduce an in-place write if it destroys a residual value still needed later.

---

# 55. MLP workspace contract

Document the real MLP path:

```text
gate projection
up projection
activation
gated product
down projection
residual
```

or whatever the frozen oracle actually uses.

Record which intermediates can safely alias after last use.

Do not fuse operations yet.

---

# 56. Output-head contract

Freeze:

```text
final normalization input/output
output projection input/output
target-token slicing rules
image reconstruction tensor shape
dtype before host image encoding
```

M1.4 must compare raw model output, not only final image bytes.

---

# 57. Dev/Base structural contract

Every new structure introduced in M1.3a must remain profile-driven.

Forbidden:

```text
if dev:
    use workspace implementation A
if base:
    use workspace implementation B
```

unless the frozen architecture truly requires distinct semantics.

Allowed differences:

```text
config values
scheduler behavior
step count
guidance/control metadata
weight revision
shape/profile-specific workspace sizes
```

The transformer implementation remains shared.

---

# 58. M1.3a.20 — Native validation after architectural refactor

After the workspace/binding/device-island refactor:

1. rerun the existing M1.3 native block fixture;
2. use the existing Python golden;
3. verify all existing internal checkpoints;
4. repeat block invocation several times;
5. inspect for device-memory growth;
6. run small compute-sanitizer coverage if practical.

Do not create new Python golden data.

---

# 59. Required M1.3a native gate

The existing block must still pass:

```text
input/output shape exact
critical internals within existing numerical contract
final block output Class D
no NaN/Inf mismatch
no leak over repeated calls
```

Additionally, prove:

```text
zero cudaMalloc/cudaFree inside block_forward
zero host tensor round trip inside block_forward
zero string weight lookup inside block_forward
```

---

# 60. Architecture inspection gate

Produce a machine- or reviewer-checkable report that lists:

```text
all allocations in model init
all allocations in workspace init
all allocations in block forward
all H2D copies in block forward
all D2H copies in block forward
all global sync calls in block forward
```

Expected hot-path result:

```text
block forward allocations: 0
block forward frees: 0
block forward H2D hidden copies: 0
block forward D2H hidden copies: 0
```

Diagnostic mode may intentionally copy selected tensors.

---

# 61. M1.3a.21 — Prepare M1.4 diagnostic checkpoints

Define, but do not yet capture, the M1.4 diagnostic points.

Use actual layer topology.

Default if homogeneous:

```text
embedding output
block 0 output
block middle output
block last output
final norm input
final norm output
final output-head result
complete model output
```

If the model has special block classes, choose checkpoints that cover transition boundaries.

---

# 62. One-oracle-run rule for M1.4

M1.3a must leave the capture tooling ready so the M1.4 Python oracle can produce all required checkpoints in **one whole-model forward**.

Do not enter M1.4 if capture tooling would require multiple full forward runs merely to obtain diagnostics.

Test capture code statically/import-only if possible.

---

# 63. Full-forward debugging strategy

Write this into `docs/M1_FORWARD_CONTRACT.md`.

When M1.4 native output differs:

```text
compare embedding
↓
compare early block
↓
compare middle block
↓
compare late block
↓
compare final norm/head
```

Then return to a local V2/V3 reproduction around the first divergent boundary.

Do not rerun the Python full model for every native change.

---

# 64. M1.3a.22 — Explicit M2/M3 backlog

Create/update:

```text
docs/M2_CANDIDATES.md
docs/M3_CANDIDATES.md
```

Only if equivalent files do not already exist.

Move non-correctness ideas there.

## M2 candidates

```text
CUDA Graphs
cuBLASLt tuning
FlashAttention/custom attention
kernel fusion
shape specialization
bandwidth optimization
scheduler device migration if measured
FP8
microbenchmarking
```

## M3 candidates

```text
pinned double-buffer loader
pread direct-to-pinned
large device arenas
soft/hard memory cap
reserve-before-allocate
async optional component prefetch
io_uring/preadv
GDS/cuFile
```

This prevents scope creep during M1.3a.

---

# 65. Things intentionally NOT implemented in M1.3a

Do not implement:

```text
full transformer forward
full denoise loop
CUDA Graph capture
FlashAttention optimization
kernel fusion
shape-specific optimized kernels
FP8/FP4
quality suite
pinned pread loader
io_uring
GDS/cuFile
production server
memory soft/hard cap
general allocator framework
```

Exception:

If an existing M1 primitive cannot execute the selected FAST_VALIDATION_RES without a structural memory failure, make the smallest correctness-preserving change and document it.

---

# 66. M1.3a commit structure

Do not squash everything into one opaque commit if the changes are substantial.

Recommended sequence:

```text
docs(m1): freeze forward topology shapes and lifetimes
```

Then:

```text
refactor(m1): bind model weights for direct layer access
```

if direct binding is not already present.

Then:

```text
refactor(m1): establish persistent device forward workspace
```

Then:

```text
test(m1): revalidate decoder block on production execution path
```

If the implementation is already partially compliant, omit redundant commits.

Do not create empty/no-op commits merely to match this list.

---

# 67. `docs/M1_FORWARD_CONTRACT.md` required sections

The document must contain:

```text
1. Oracle revision
2. Dev/Base model revisions
3. Transformer topology
4. Block topology
5. Input contract
6. Output contract
7. Sequence-length formulas
8. 1024 shape profile
9. 2048 shape profile
10. GQA contract
11. MRoPE contract
12. Residual contract
13. MLP contract
14. Output-head contract
15. Scheduler boundary
16. CPU/GPU ownership map
17. Allowed synchronization points
18. M1.4 diagnostic checkpoint plan
```

---

# 68. `docs/M1_MEMORY_LIFETIMES.md` required sections

Include:

```text
1. Buffer inventory
2. Size formula for each buffer
3. Lifetime class
4. Birth / last use
5. Read/write ownership
6. Alias candidates
7. Ping-pong plan
8. Profile-specific maximum sizes
9. Simultaneous-live peak
10. 1024 theoretical peak
11. 2048 theoretical peak
12. Reference attention O(S²) warning if applicable
```

Use diagrams where useful.

---

# 69. `docs/M1_EXECUTION_ARCHITECTURE.md` required sections

Include:

```text
1. Single semantic execution path
2. Model init
3. Weight binding
4. Workspace init
5. Full forward device island
6. Diagnostic hooks
7. Scheduler boundary
8. Error propagation
9. Stream/handle ownership
10. Resource cleanup
11. Persistent-process readiness
12. CUDA Graph compatibility constraints
13. Forbidden hot-path operations
```

---

# 70. Error handling requirements

All new init/binding/workspace operations must fail with actionable messages.

Include:

```text
profile
layer
tensor
expected shape
actual shape
requested bytes
buffer category
CUDA/cuBLAS error
```

where applicable.

Do not convert a binding/allocation failure into undefined behavior later in the forward.

---

# 71. Cleanup requirements

Test:

```text
model init -> destroy
model init -> workspace init -> destroy
model init -> workspace init -> block fixture -> destroy
repeated block fixture -> destroy
```

Verify cleanup remains deterministic.

Do not require a full model forward.

---

# 72. Repository hygiene

Before each M1.3a commit:

```bash
git status --short
git diff --check
git diff
```

After:

```bash
git status --short
git log -1 --oneline
```

Do not commit:

```text
/python
/models
large diagnostic dumps
golden binaries
build output
```

Update:

```text
docs/M1_STATUS.md
```

after each sub-gate.

---

# 73. M1.3a stop conditions

Stop rather than proceed to M1.4 if:

```text
M1.3 parity regresses
a block still allocates device memory internally
hidden state crosses device->host->device between blocks
runtime still performs string tensor lookup per layer
real sequence length cannot be derived unambiguously
1024/2048 workspace sizing cannot be computed
residual lifetime cannot be proven safe with ping-pong
block classes differ but the runtime assumes one topology
Base support would require a separate transformer implementation
M1.4 capture cannot collect diagnostic checkpoints in one oracle forward
```

Fix the structural issue at M1.3a level.

Do not move forward and plan to repair it after M1.4.

---

# 74. M1.3a gate checklist

M1.3a is green only when all are true:

```text
[ ] M1.3 remains green
[ ] real T2I call graph frozen
[ ] block classes identified
[ ] real target token/sequence formulas frozen
[ ] DEV-FAST shape profile frozen
[ ] DEV-1024 shape profile frozen
[ ] DEV-2048 shape profile frozen
[ ] tensor binding is direct after init
[ ] no string lookup in block hot path
[ ] CPU/GPU ownership map complete
[ ] memory/lifetime diagram complete
[ ] hidden ping-pong plan complete
[ ] reusable Q/K/V workspace defined
[ ] reusable attention workspace defined
[ ] reusable MLP workspace defined
[ ] output workspace defined
[ ] theoretical 1024 memory requirement calculated
[ ] theoretical 2048 memory requirement calculated
[ ] no cudaMalloc/cudaFree in block hot path
[ ] no malloc/free in block hot path
[ ] no block-to-block D2H/H2D hidden transfers
[ ] no weight file I/O in forward path
[ ] no runtime weight repack/transpose in per-step path
[ ] persistent CUDA/cuBLAS resources have defined ownership
[ ] scheduler boundary documented
[ ] diagnostic hooks use same semantic path
[ ] M1.4 checkpoint capture plan prepared
[ ] one Python M1.4 forward will be enough for all checkpoints
[ ] CUDA Graph-incompatible dynamic patterns avoided where practical
[ ] existing M1.3 native block golden still passes
[ ] repeated native block calls show no memory growth
[ ] Dev/Base remain one transformer implementation
[ ] M2/M3 optimizations explicitly deferred
[ ] repository clean
```

---

# 75. Exit state

When the checklist is green, update:

```text
docs/M1_STATUS.md
```

with:

```text
M1.3a COMPLETE
M1.4 READY
```

Include:

```text
closing commit
selected FAST_VALIDATION_RES
DEV-1024 sequence length
DEV-2048 sequence length
workspace bytes by profile
number of persistent device buffers
number of allocations in block forward = 0
number of hidden D2H/H2D transfers between blocks = 0
```

---

# 76. Exact next action after M1.3a

Only then begin M1.4.

M1.4 must:

1. perform the planned single Python whole-model forward;
2. capture all prepared checkpoints in that one run;
3. execute the native complete transformer forward using the M1.3a device-resident architecture;
4. debug against the frozen checkpoints;
5. never rebuild a temporary host-resident full-forward path.

---

# 77. Agent execution order

Follow this order exactly:

```text
A. Confirm M1.3 gate green
B. Read frozen oracle forward statically
C. Create M1_FORWARD_CONTRACT.md
D. Resolve target/image token and sequence formulas
E. Freeze DEV-FAST / DEV-1024 / DEV-2048 shape inventory
F. Classify any distinct transformer block types
G. Audit current tensor lookup/binding
H. Bind weights to stable layer structures if needed
I. Create M1_MEMORY_LIFETIMES.md
J. Calculate simultaneous-live workspace requirements
K. Define hidden A/B ping-pong
L. Define reusable Q/K/V, attention and MLP workspace
M. Audit allocations/copies/synchronizations in current block path
N. Remove hot-path allocations and block-to-block host transfers
O. Create M1_EXECUTION_ARCHITECTURE.md
P. Establish diagnostic hooks on the same semantic path
Q. Prepare M1.4 one-run checkpoint capture
R. Rerun existing native M1.3 block fixture only
S. Verify no leak/memory growth
T. Update M1_STATUS
U. Commit M1.3a closure
V. Begin M1.4 only when every gate is green
```

---

# 78. Guiding architecture

The intended pre-M1.4 execution structure is:

```text
MODEL INIT
    │
    ├─ parse profile/config
    ├─ load weights
    ├─ bind tensors once
    ├─ create CUDA/cuBLAS resources once
    └─ allocate persistent workspace once
              │
              ▼
REQUEST
    │
    ├─ frozen/pre-tokenized input
    ├─ request/image state
    └─ timestep
              │
              ▼
        DEVICE-RESIDENT FORWARD
              │
              ├─ embeddings / target projection
              │
              ├─ hidden_A
              │      ↓
              ├─ block 0 -> hidden_B
              │      ↓
              ├─ block 1 -> hidden_A
              │      ↓
              ├─ ...
              │      ↓
              ├─ final norm
              │
              └─ output head
              │
              ▼
         raw device output
              │
              ▼
       scheduler boundary
```

No intermediate CPU detour is part of normal transformer execution.

---

# 79. Final principle

M1.3a is successful if M1.4 becomes mostly a **composition problem**, not an architecture redesign.

After this milestone:

- the model topology is known;
- the shapes are known;
- the memory lifetimes are known;
- weights are already bound;
- workspaces already exist;
- the block chain is already device-resident by construction;
- diagnostics already know where to observe;
- the scheduler boundary is explicit;
- the first full forward can be assembled without creating temporary architectural debt.

> **The first complete forward should already have the ownership, lifetime and execution structure that production will keep. Optimization comes later; structural mistakes do not.**
