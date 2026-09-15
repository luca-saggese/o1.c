# M3 — GB10 Production Loader, Memory Residency and Execution Pipeline

## Status

Proposed milestone after M2.

## Purpose

M3 converts the correct and performance-characterized HiDream-O1 native runtime into a production-grade GB10 execution architecture focused on:

- startup/load latency;
- bounded memory behavior on GB10 unified memory;
- persistent resident model state;
- elimination of avoidable allocation/synchronization overhead;
- long device-side execution islands;
- asynchronous I/O only where it improves end-to-end wall time;
- stable, measurable behavior under repeated inference.

M3 is **not** another correctness milestone and is **not** a license to rewrite working M1/M2 paths.

The M1 numerical contract remains authoritative. The M2 profiler/baseline determines which M3 optimizations are worth enabling.

---

# 1. Prior GB10 evidence carried into M3

A previous GB10 inference engine showed a large startup improvement after replacing a giant synchronous pageable/file-backed → CUDA copy with bounded double-buffered pinned staging:

```text
file / storage
    ↓
pinned stage A/B
    ↓
cudaMemcpyAsync
    ↓
final persistent CUDA allocation
```

The operating pattern was:

```text
CPU fills stage[1]
while
GPU copies stage[0]

then swap
```

with:

```text
2 × fixed-size pinned staging buffers
no cudaMalloc per chunk
no synchronization per chunk
CUDA events only for stage-slot reuse
one synchronization at the end of the load phase
```

That previous engine observed about `432 s -> 83 s` startup. This is evidence that the pattern is worth testing on GB10, **not** a target for HiDream.

HiDream must establish its own baseline.

---

# 2. Decision matrix

| Prior GB10 pattern | Decision for HiDream | Rationale |
|---|---|---|
| Double pinned staging + `cudaMemcpyAsync` | **APPLY** | High-value, low-complexity startup optimization |
| `pread()` directly into pinned stage | **APPLY** | Preferred when file offsets are already known |
| Fixed bounded staging | **APPLY** | Must not scale with largest tensor |
| 2 staging slots | **APPLY FIRST** | Enough to overlap blocking read with async copy in most cases |
| Many small CUDA allocations | **REJECT** | Replace with a few persistent arenas |
| Exactly 5 CUDA regions | **DO NOT COPY LITERALLY** | Region count follows HiDream lifetime/layout, not q38 history |
| Persistent workspaces | **APPLY** | No `cudaMalloc/cudaFree` in denoising hot path |
| Long device island | **APPLY STRONGLY** | Iterative denoising magnifies host round-trip cost |
| No resident dequantized duplicate | **CONDITIONAL APPLY** | Relevant only if M2 adds quantized/reduced-precision modes |
| Shape-specialized kernels | **PROFILE-GATED** | Specialize only real hot shapes |
| Single semantic execution path | **APPLY** | Prevent backend/device divergence |
| Bundle-level measurement | **APPLY** | Kernel-only wins are insufficient |
| Reserve before allocation | **APPLY** | Important on 128 GiB shared CPU/GPU memory |
| Soft + hard watermark | **APPLY, RE-DERIVE VALUES** | Do not copy 114/118 GiB blindly |
| No giant temporary buffers | **APPLY** | Staging/workspaces stay bounded |
| Async I/O with compute | **CONDITIONAL** | Useful for genuinely optional/file-backed components |
| Aggressive core unload/reload | **REJECT** | Core transformer stays resident |
| `io_uring` / `preadv` | **DEFER UNTIL MEASURED** | Add only if `pread` remains submission/syscall-bound |
| GDS / cuFile | **DEFER UNTIL MEASURED** | More complexity; benchmark only if simple path remains bottleneck |
| NUMA/pinned placement tuning | **DEFER / PROFILE-GATED** | Only after CPU-side loader bottleneck is proven |

---

# 3. HiDream-specific residency model

The generic T2I split:

```text
text encoder
VAE
UNet/DiT
ControlNet
LoRA
```

must not be copied blindly.

HiDream-O1 uses a unified transformer-style generation path, so residency must be derived from the **actual frozen tensor manifest and T2I execution graph**.

## Resident for the full generation

```text
selected profile core transformer weights
input/patch projection used by T2I
timestep/conditioning projections
all transformer block weights
final norm/output/pixel projection
scheduler state
current denoising state
persistent activation arenas
persistent GEMM/attention workspace
small device constants/tables
```

## Host-resident only

```text
tokenizer vocabulary / merges / metadata
model manifest
tensor directory / file descriptors during load
CLI/configuration
small diagnostic metadata
```

## Potentially load-on-demand

Only after graph inspection proves they are unused by normal T2I:

```text
optional adapters
LoRA modules
reference-image-only components
vision/reference submodules unused by pure T2I
alternate optional heads
```

Do not classify a subtree as optional merely from its name.

## Profile residency

Keep only the active profile resident:

```text
Dev OR Base
```

not both simultaneously.

Profile switching may unload/reload the model because it is outside the denoising hot path.

---

# 4. Core invariants

1. The active core transformer remains resident for the full sampling loop.
2. No `cudaMalloc` or `cudaFree` in the per-step hot path.
3. No file I/O in the per-step core path.
4. No avoidable host round trip between transformer blocks.
5. Weight loading uses bounded memory.
6. Staging size is fixed/configurable and independent of largest tensor.
7. Final device regions are allocated before streamed copy begins.
8. Memory budget is reserved before allocation.
9. Loader synchronization is event-driven, not per-chunk global synchronization.
10. M3 changes preserve the M1 numerical contract.
11. M3 performance is measured at bundle/process level.
12. There is one semantic execution path; diagnostics wrap it.

---

# 5. What M3 must not do

M3 must not:

- introduce a separate “fast” transformer with duplicate semantics;
- trade correctness for startup speed;
- keep a full dequantized mirror of quantized weights for convenience;
- size staging from the largest tensor;
- allocate/free device memory per tensor/chunk;
- synchronize globally after every chunk;
- evict/reload core weights between sampling steps;
- assume unified physical memory makes all copy strategies equivalent;
- adopt GDS/cuFile merely because it exists;
- accept a faster loader if complete process-to-ready time regresses;
- copy the q38 114/118 GiB thresholds without re-deriving them.

---

# 6. M3 validation levels

Reuse M1/M2 correctness artifacts wherever possible.

| Level | Purpose |
|---|---|
| P0 | no model math: allocator/load-plan/unit tests |
| P1 | model load/startup only |
| P2 | one native forward using existing M1 golden |
| P3 | native 3-step denoising using existing M1 golden |
| P4 | full native Dev generation for closure only |

Python oracle reruns are normally unnecessary in M3.

A loader change should normally be validated:

```text
P0 -> P1 -> P2
```

not with a full generation.

---

# 7. M3 benchmark contract

Before changing loader/allocator, freeze a new operational baseline.

Measure at least:

```text
process start -> manifest parsed
manifest parsed -> final device arenas allocated
weight-load start -> final copy complete
model ready -> first-forward ready
one forward
3-step run
steady-state full Dev generation
peak host RSS
peak pinned staging bytes
peak CUDA-owned bytes
peak accounted bytes
```

Record both:

```text
cold filesystem cache
warm filesystem cache
```

where practical.

For startup-only benchmarks:

```text
3 runs
median
min
max
```

No full image generation is needed to benchmark startup.

---

# 8. Unified-memory accounting model

GB10 CPU and GPU share the 128 GiB physical memory pool.

M3 must therefore account for:

```text
resident_weights
persistent_activations
persistent_workspace
scheduler_state
optional_module_reserve
pinned_staging
host_metadata
safety_reserve
```

as parts of one process-level memory plan.

Do not reason about host and CUDA allocations as if they were independent physical capacity pools.

---

# 9. Soft and hard limits

Implement configurable:

```text
soft_limit
hard_limit
```

## Soft limit

Crossing it:

- emits structured diagnostics;
- prevents optional prefetch that would reduce safety margin;
- may reclaim truly optional cached resources;
- never unloads the active core during generation.

## Hard limit

Any reservation that would exceed it fails **before** CUDA allocation.

Report:

```text
requested bytes
currently reserved bytes
hard limit
allocation category
profile
resolution
batch/guidance mode
```

---

# 10. Do not hardcode 114/118 GiB

The previous engine's `114/118 GiB` thresholds were appropriate to its own ~72.5 GiB resident layout.

M3 must derive HiDream defaults from:

```text
actual resident weights
worst validated activation/workspace requirement
pinned staging requirement
OS/driver/runtime headroom
optional reserve
```

Requirements:

```text
soft < hard
hard < total unified physical memory
explicit system safety reserve remains
```

Expose overrides, but document safe defaults.

---

# 11. Reserve-before-allocate

Conceptually:

```text
reserve(bytes, category)
    ↓
if reserved + bytes > hard_limit:
    fail
    ↓
cudaMalloc / cudaMallocHost
    ↓
commit reservation
```

On allocation failure:

```text
release reservation
return structured error
```

On free:

```text
release accounting
```

If concurrent loader/prefetch threads allocate, reservation must be thread-safe.

---

# 12. Device arena strategy

Target:

```text
few large persistent allocations
```

not one allocation per tensor, and not “exactly five”.

Illustrative layout:

```text
Arena 0: resident model weights
Arena 1: persistent activation ping/pong
Arena 2: attention/GEMM workspace
Arena 3: scheduler/output/small persistent state
Arena 4: optional-module arena, if actually needed
```

Split by dtype/alignment only when it simplifies real kernels.

Final report must show:

```text
region count
purpose
size
alignment
lifetime
```

---

# 13. Tensor pointer mapping

Every tensor resolves to:

```text
arena_id
offset
size
dtype
shape
layout
```

Validate host-side:

```text
alignment
bounds
no overlap
byte size
```

No individual device allocation should be required for every weight tensor.

---

# 14. Preferred loader architecture

For regular local model files with known offsets:

```text
NVMe / filesystem
      ↓
pread(fd, pinned_stage[slot], n, file_offset)
      ↓
cudaMemcpyAsync(final_device_ptr, pinned_stage[slot], n, H2D, copy_stream)
      ↓
cudaEventRecord(slot_done)
      ↓
reuse slot only after its event completes
```

Initial design:

```text
2 staging slots
fixed CHUNK
one copy stream
one event per slot
one final stream synchronization
```

No synchronization per tensor/chunk except slot-reuse protection.

---

# 15. `pread()` versus `mmap -> memcpy -> pinned`

For sequential one-shot loading, if the model format already gives `fd + offset + length`, prefer:

```text
pread -> pinned stage -> CUDA
```

over:

```text
mmap -> CPU memcpy -> pinned stage -> CUDA
```

because it:

- removes a large avoidable CPU-side copy;
- bounds host staging;
- makes I/O accounting explicit;
- maps well to sharded weight files.

Keep mmap only as fallback/debug path if useful.

---

# 16. Loader state machine

```text
PLAN
  ↓
RESERVE
  ↓
ALLOCATE_FINAL_REGIONS
  ↓
OPEN_FILES
  ↓
ALLOCATE_PINNED_STAGES
  ↓
STREAM_CHUNKS
  ↓
FINAL_SYNC
  ↓
VERIFY
  ↓
FREE_STAGING
  ↓
READY
```

Any failure must unwind owned resources.

Never mark the model READY after partial load.

---

# 17. Reference double-buffer flow

```c
allocate_final_device_regions();

cudaMallocHost(&stage[0], CHUNK);
cudaMallocHost(&stage[1], CHUNK);

for each planned source span {
    while (remaining) {
        slot = chunk_index & 1;

        if (slot_was_used(slot))
            wait_for_slot_event(slot);

        n = min(CHUNK, remaining);

        pread_exact(fd, stage[slot], n, file_offset);

        cudaMemcpyAsync(
            final_device_base + device_offset,
            stage[slot],
            n,
            cudaMemcpyHostToDevice,
            copy_stream);

        cudaEventRecord(done[slot], copy_stream);

        file_offset   += n;
        device_offset += n;
        remaining     -= n;
        chunk_index++;
    }
}

cudaStreamSynchronize(copy_stream);

verify_loader_state();

destroy_events();
cudaFreeHost(stage[0]);
cudaFreeHost(stage[1]);
```

Rules:

- slot wait is specific to that slot, not a device-global sync;
- no `cudaStreamSynchronize()` per chunk;
- no device allocation in chunk loop;
- `pread_exact()` handles short reads and `EINTR`.

---

# 18. Chunk size

Start benchmarking around:

```text
128 MiB
```

because it worked well in previous GB10 experience, but treat it only as an initial hypothesis.

Small fixed sweep:

```text
32 MiB
64 MiB
128 MiB
256 MiB
```

Measure startup only:

```text
cold load time
warm load time
read GB/s
copy GB/s
copy-stream idle
read-side idle
peak pinned bytes
```

Do not autotune every process startup.

For two 128 MiB slots:

```text
pinned staging = 256 MiB
```

bounded and predictable.

---

# 19. Load plan and coalescing

Build the source plan from:

```text
file
file offset
length
destination arena
destination offset
```

For each shard:

1. sort required spans by file offset;
2. preserve sequential access where practical;
3. coalesce adjacent source spans when destination layout permits;
4. avoid pathological tiny reads.

Report:

```text
source file count
raw tensor spans
coalesced spans
total bytes
seek/discontinuity count
```

---

# 20. Packed deployment layout — conditional

If M1/M2 already has a packed native format, optimize its physical ordering.

If runtime still reads safetensors directly, evaluate packing only if M3.2 shows meaningful overhead from:

```text
many shards
fragmented offsets
metadata parsing
runtime repacking/transposes
```

Ideal pack:

```text
small header
tensor directory
region descriptors
weight bytes in final arena-load order
```

Packing is offline and must not add a Python runtime dependency.

---

# 21. Allocate final regions before copying weights

Before large payload copy, reserve/allocate at least:

```text
weights
minimum activation arenas
minimum persistent workspace
scheduler/output state
```

This gives:

- fail-fast memory behavior;
- stable final addresses;
- no allocator fragmentation during load;
- direct copy-plan destinations.

Do not discover workspace allocation failure only after a 30+ GiB model load.

---

# 22. Persistent workspace

After READY:

```text
cudaMalloc/cudaFree in normal sampling path == forbidden
```

Pre-plan workspace by execution profile:

```text
resolution
batch
guidance mode
precision
```

Workspace may be rebuilt/resized before inference begins, never once per diffusion step.

---

# 23. Activation reuse

Where lifetimes permit, use bounded:

```text
A
B
scratch
```

rather than one allocation per layer.

Lifetime analysis must prove reuse safety.

Any change must preserve M1 numerical parity.

---

# 24. Long device island

Target:

```text
host:
    parse prompt
    tokenize
    prepare small metadata
    launch inference

device:
    image/noise state
    timestep conditioning
    transformer block 0
    ...
    transformer block N
    scheduler update
    next step
    ...
    final raw image

host:
    copy final result
    encode/save
```

Avoid:

```text
block -> D2H -> CPU operation -> H2D -> block
```

unless algorithmically required.

---

# 25. Scheduler residency

If M2 still performs scheduler arithmetic on CPU, M3 should **measure** whether it creates a real critical-path synchronization.

If yes, consider device-resident scheduler updates.

If not, keep the simpler implementation.

Validate against existing M1 V4 goldens.

---

# 26. Single semantic path

Avoid:

```text
backend_forward()
device_forward()
fast_forward()
debug_forward()
```

Prefer one core path:

```text
forward(execution_plan, instrumentation_flags)
```

Instrumentation may dump/timestamp without changing semantics.

Apply same rule to:

```text
scheduler
weight mapping
tokenizer
profile selection
```

---

# 27. Reduced-precision residency policy

Only relevant if M2 introduced a reduced-precision/quantized mode.

If kernels consume resident low-precision weights directly, keep them in that native packed format.

If explicit dequantization is needed, prefer:

```text
quantized resident weights
    ↓
fused/on-the-fly decode
    ↓
compute
```

Avoid:

```text
quantized resident copy
+
full resident BF16 mirror
```

unless benchmarked and explicitly memory-budgeted.

For BF16 mode there is no dequantization issue:

```text
file BF16 -> pinned stage -> resident BF16 arena
```

---

# 28. Shape specialization

Apply only when M2/M3 profiling proves:

```text
significant wall-time share
stable/few real shapes
generic kernel inefficiency
```

Candidate dimensions may include:

```text
hidden size
head dim
Q heads
KV heads
MLP width
image-token count
CFG batch
```

Keep a generic fallback.

Accept specialization only if:

1. numerical parity passes;
2. enclosing block/forward gets faster;
3. no harmful new sync appears;
4. unsupported shapes fall back cleanly.

Kernel microbenchmark wins alone do not count.

---

# 29. Optional module streaming

Core weights stay resident.

Only genuinely optional components may stream.

Policy:

```text
request optional component
    ↓
reserve memory
    ↓
start file read/copy early
    ↓
continue independent compute
    ↓
wait only at first actual use
```

Measure:

```text
prefetch start
I/O finish
first-use point
critical wait
total wall
```

Success means the first-use stall is materially reduced/hidden.

---

# 30. Never stream the active core transformer

Do not use a block-by-block load/execute/unload design on current HiDream/GB10 if the active core fits resident memory.

Denoising reuses the same weights every step; repeated NVMe traffic would dominate.

---

# 31. `io_uring`, `preadv`, GDS/cuFile, NUMA

Start with:

```text
pread + 2 pinned slots
```

Only progress to `preadv`/`io_uring` if syscall/submission overhead remains material.

Evaluate GDS/cuFile only if:

- software/filesystem/NVMe support is solid;
- pinned pread remains materially I/O/copy bound;
- deployment complexity is acceptable;
- complete process-to-ready improves.

Tune NUMA/CPU placement only after a CPU-side loader bottleneck is measured.

---

# 32. M3.0 — Freeze operational baseline

## Work

Create:

```text
docs/M3_STATUS.md
docs/M3_BASELINE.md
artifacts/m3/
```

Record:

```text
engine commit
GB10 identity
CUDA version
model profile/revision
precision
validation resolution
performance resolution
startup loader architecture
allocation count
resident weight bytes
workspace bytes
peak memory
startup phase timings
```

Collect:

```text
cold model-load median
warm model-load median
process-to-ready median
one-forward time
3-step time
full Dev time from M2 report
```

No implementation change yet.

## Gate M3.0

Baseline reproducible enough to distinguish wins/regressions.

## Commit

```text
bench(m3): freeze gb10 startup and memory baseline
```

---

# 33. M3.1 — Memory planner, hard cap and persistent arenas

## Work

Implement:

```text
memory categories
soft limit
hard limit
reserve-before-allocate
persistent weight arena(s)
persistent activation/workspace arenas
allocation map/report
```

## Validation

P0:

```text
arena layout
overflow
alignment
reservation rollback
hard-limit rejection
concurrent reservation if applicable
```

P1:

```text
full startup/load
```

P2:

```text
one native forward against existing M1 golden
```

## Gate M3.1

```text
no per-tensor cudaMalloc
no hot-path cudaMalloc/cudaFree
hard cap enforced before allocation
soft watermark works
all tensor mappings in bounds
M1 parity preserved
```

## Commit

```text
perf(m3): add bounded gb10 memory planner and persistent arenas
```

---

# 34. M3.2 — Double-buffered pinned `pread` loader

## Work

Implement:

```text
load plan
2 pinned staging buffers
pread_exact
copy stream
slot completion events
one final stream synchronization
structured timing
```

Initial default candidate:

```text
2 × 128 MiB
```

but keep CHUNK configurable.

## Timings

Record:

```text
open/metadata
arena allocation
pread wall
async-copy submission
final sync wait
total weight load
total process-to-ready
```

## Validation

P0:

```text
short read
EINTR
EOF
offset correctness
slot reuse
event ordering
```

P1:

```text
model startup
selected weight fingerprints
```

P2:

```text
one-forward parity
```

## Gate M3.2

```text
bounded pinned memory
no chunk cudaMalloc/free
no per-chunk global sync
all weight fingerprints correct
process-to-ready improves versus M3.0
OR trace proves loader is no longer material bottleneck
```

No q38-equivalent speedup is required.

## Commit

```text
perf(m3): stream weights through double-buffered pinned staging
```

---

# 35. M3.3 — Load-plan coalescing / deployment layout

Entry only if M3.2 shows meaningful remaining source-side overhead.

Candidate changes:

```text
sort by file offset
coalesce adjacent spans
reduce syscall count
pack weights in final arena order
remove runtime transpose/repack
```

If not justified:

```text
M3.3 NOT NEEDED
```

with evidence.

## Commit if implemented

```text
perf(m3): coalesce sequential model load spans
```

or:

```text
perf(m3): pack model in resident arena load order
```

---

# 36. M3.4 — Persistent workspace and long device island

Audit every per-step operation:

```text
operation
host/device
allocation?
H2D?
D2H?
sync?
required?
```

Remove avoidable allocation/copy/synchronization.

## Validation

P2:

```text
one-forward parity
```

P3:

```text
3-step parity using existing M1 golden
```

Benchmark complete 3-step bundle.

## Gate M3.4

```text
zero cudaMalloc/cudaFree inside normal denoising step
zero file I/O inside normal denoising step
no avoidable block-to-block host round trip
3-step parity preserved
3-step wall does not regress
```

## Commit

```text
perf(m3): keep denoising state and workspaces device-resident
```

---

# 37. M3.5 — Quantized residency policy

Run only if M2 introduced reduced-precision/quantized execution.

Otherwise:

```text
M3.5 N/A
```

Record for each tensor class:

```text
on-disk format
resident format
kernel-consumed format
scale metadata
temporary decode requirement
```

Gate:

```text
no accidental full BF16 duplicate
memory planner includes scales/metadata
numerical contract remains green
complete bundle does not regress
```

## Commit

```text
perf(m3): keep reduced-precision weights in native resident format
```

---

# 38. M3.6 — Profile-gated shape specialization

For each candidate:

1. freeze enclosing-operation baseline;
2. implement one specialization;
3. verify parity;
4. benchmark kernel;
5. benchmark block;
6. benchmark forward/3-step bundle;
7. accept only if bundle improves.

## Commit pattern

```text
perf(m3): specialize <operation> for hidream <shape>
```

One specialization per commit.

---

# 39. M3.7 — Optional component prefetch

Run only if there is a real optional file-backed component in the supported product path.

Do not invent one just to satisfy M3.

Gate measures:

```text
prefetch lead time
I/O duration
wait at first use
total generation wall
```

If none exists:

```text
M3.7 N/A
```

---

# 40. M3.8 — Production closure

Minimum matrix:

```text
Dev cold startup
Dev warm startup
Dev one forward
Dev 3-step
Dev full native generation
second Dev generation without reload
load -> generate -> destroy
repeated load -> destroy
Base startup/load
Base one-forward compatibility
```

Do not run a 50-step Base generation unless separately required.

Second generation in same process must not:

- reload core weights;
- reallocate core arenas;
- grow persistent memory;
- rebuild immutable metadata;
- recreate large workspaces unnecessarily.

Record first/second generation time and memory high-water marks.

---

# 41. Startup acceptance

M3 closure must show:

1. startup/load wall time versus M3.0;
2. phase breakdown explaining the change;
3. bounded staging memory;
4. stable resident memory;
5. no numerical regression.

Do not set a fake universal target such as `83 s`.

Report both absolute and relative improvement.

---

# 42. Memory acceptance

Report:

```text
model file bytes
resident weight bytes
persistent activation bytes
persistent workspace bytes
pinned staging bytes
peak accounted bytes
peak observed memory
soft limit
hard limit
system safety reserve
```

Required:

```text
no hard-limit oversubscription
no hidden giant temporary allocation
no unbounded staging growth
no persistent memory growth across repeated generations
```

---

# 43. Bundle-level performance rule

Each performance commit must include an enclosing metric.

Loader change:

```text
process-to-ready
```

Workspace/device-island change:

```text
3-step wall
one-forward wall
```

Kernel specialization:

```text
block wall
one-forward wall
```

Never accept:

```text
kernel faster
but total wall slower
```

---

# 44. Instrumentation

Prefer:

```text
NVTX ranges
CUDA events
monotonic CPU timers
structured JSON timings
```

Instrumentation must not introduce synchronization that destroys the overlap being measured.

---

# 45. Loader integrity and errors

Detect/report:

```text
open failure
file-size mismatch
invalid tensor offset
short read
unexpected EOF
pread error
CUDA copy error
event error
reservation failure
CUDA allocation failure
manifest mismatch
fingerprint mismatch
```

Partial load must never become READY.

After async load, verify selected weights from early/middle/late/head regions before any forward.

This catches:

```text
wrong offset
stage reuse race
copy overlap bug
destination offset bug
short-read bug
```

---

# 46. File lifecycle

For fully resident core weights:

```text
open
stream
verify
close
```

After READY, core inference should not require source model files.

Optional streamable modules use separate lifecycle.

---

# 47. I/O worker model

Start with:

```text
one CPU loader thread
one CUDA copy stream
two pinned slots
```

Because `pread()` blocks the CPU while the previous DMA can continue, useful overlap already exists.

Do not add a thread pool until measured necessary.

---

# 48. Cold/warm cache reporting

Report cold and warm separately.

Do not present a warm page-cache load as cold startup.

Annotate likely cache state.

Privileged cache dropping is not required for ordinary developer tests.

---

# 49. M3 report

Create:

```text
docs/M3_REPORT.md
```

Required sections:

## Revisions

```text
M3 closing commit
M2 baseline commit
oracle revision
Dev revision
Base revision
```

## Loader

```text
source format
load-plan type
stage count
chunk size
copy stream count
event strategy
final synchronization count
```

## Memory

```text
arena map
soft limit
hard limit
system reserve
peak values
```

## Startup comparison

```text
metric                 M3.0 baseline    M3 final    delta
cold model load
warm model load
process-to-ready
allocation phase
I/O phase
final sync wait
```

## Runtime comparison

```text
one forward
3-step
full Dev
second warm generation
```

## Correctness

State which M1 goldens/gates were revalidated.

## Conditional features

Explicitly report:

```text
packed deployment layout: used / not needed
quantized resident format: used / N/A
shape specialization: list / not needed
optional async prefetch: used / N/A
io_uring: used / not justified
GDS/cuFile: used / not justified
```

## Remaining bottlenecks

Rank by measured complete-wall impact.

---

# 50. Canonical commit sequence

```text
bench(m3): freeze gb10 startup and memory baseline

perf(m3): add bounded gb10 memory planner and persistent arenas

perf(m3): stream weights through double-buffered pinned staging

perf(m3): coalesce sequential model load spans
    # conditional

perf(m3): keep denoising state and workspaces device-resident

perf(m3): keep reduced-precision weights in native resident format
    # conditional

perf(m3): specialize <operation> for hidream <shape>
    # conditional; one commit per specialization

perf(m3): prefetch optional model components before first use
    # conditional

test(m3): validate production gb10 residency and startup pipeline
```

Do not create commits for conditional work that was not needed.

---

# 51. Gate summary

## M3.0

```text
operational baseline frozen
startup phase timings reproducible
memory baseline known
```

## M3.1

```text
bounded memory planner
reserve-before-allocate
few persistent arenas
no hot-path allocation
```

## M3.2

```text
pread -> pinned A/B -> cudaMemcpyAsync -> resident arena
bounded staging
no per-chunk allocation
no per-chunk global sync
weight integrity preserved
startup improved or proven non-bottleneck
```

## M3.3

```text
coalescing/packing improves startup
or N/A with evidence
```

## M3.4

```text
persistent workspace
long device island
3-step parity
no allocation/file-I/O in normal step
```

## M3.5

```text
quantized residency without duplicate expansion
or N/A
```

## M3.6

```text
only profiler-proven shape specialization
bundle wall improves
```

## M3.7

```text
optional prefetch hides real critical stall
or N/A
```

## M3.8

```text
full Dev native run passes
second run reuses resident core
memory stable
startup/memory report complete
Base load/forward compatibility preserved
```

---

# 52. Definition of M3 DONE

```text
[ ] M3.0 baseline frozen
[ ] active profile uses few persistent device regions
[ ] reserve-before-allocate hard cap enforced
[ ] soft watermark implemented
[ ] limits are HiDream-derived, not copied from q38
[ ] core loader uses bounded pinned staging
[ ] preferred local path uses pread when applicable
[ ] staging uses fixed-size slots
[ ] no cudaMalloc/cudaFree in chunk loop
[ ] no per-chunk global synchronization
[ ] core transformer remains resident across sampling steps
[ ] no file I/O in normal denoising hot path
[ ] persistent workspace is reused
[ ] repeated generation does not reload core weights
[ ] repeated generation does not show persistent memory growth
[ ] M1 numerical contract remains green
[ ] startup wall is compared against M3.0
[ ] bundle metrics accompany kernel-level claims
[ ] Base profile load/forward compatibility remains green
[ ] docs/M3_REPORT.md complete
[ ] engine repository clean
```

Conditional items may be marked N/A with evidence.

---

# 53. Recommended agent order

```text
A. Read M1/M2 reports and current WORKING_MODE
B. Create M3_STATUS and M3_BASELINE
C. Freeze M3.0 startup/memory baseline
D. Map current CUDA allocations and tensor lifetimes
E. Implement memory planner + limits + persistent arenas
F. Validate P0/P1/P2
G. Implement two-slot pinned pread loader
H. Sweep small fixed chunk-size set using startup-only runs
I. Select/freeze default chunk size
J. Validate weight fingerprints before model math
K. Validate one forward against existing M1 golden
L. Decide whether load-plan packing/coalescing is justified
M. Implement only if measured
N. Audit hot-path allocations/copies/synchronizations
O. Build persistent workspace/device island
P. Validate 3-step numerical parity
Q. Apply quantized-residency policy only if relevant
R. Apply shape specialization only where profiler proves value
S. Apply optional async prefetch only for real optional modules
T. Run M3.8 production closure
U. Write M3_REPORT
V. Commit closure
W. STOP
```

---

# 54. Final architectural target

```text
                    PROCESS START
                         │
                         ▼
                parse small metadata
                         │
                         ▼
               build memory/load plan
                         │
                         ▼
              reserve memory budget
                         │
                         ▼
          allocate final persistent regions
                         │
                         ▼
       ┌─────────────────────────────────┐
       │ fixed pinned stage A / stage B │
       └─────────────────────────────────┘
             ▲                    │
             │                    ▼
         pread NVMe        cudaMemcpyAsync
             │                    │
             └──── overlap ───────┘
                                  │
                                  ▼
                     resident CUDA weight arenas
                                  │
                                  ▼
                        persistent workspaces
                                  │
                                  ▼
              ┌─────────────────────────────┐
              │ long device-side T2I island │
              │ transformer                 │
              │ scheduler/state             │
              │ transformer                 │
              │ scheduler/state             │
              │ ...                         │
              └─────────────────────────────┘
                                  │
                                  ▼
                         final image output
```

Central M3 principle:

> **Pay storage/allocation costs once, bound them explicitly, keep the active model resident, and keep the iterative image-generation path on device for as long as the algorithm allows.**
