# M1-post — Full HiDream-O1-Image Feature Parity

## Status

This milestone is executed **after M1 correctness closure and before M2 performance optimization**.

M1 proved that the native C/CUDA transformer, scheduler, tokenizer, Dev path, Base/Full path, and selected end-to-end generation semantics can reproduce the frozen Python oracle.

M1-post exists because a correct internal validation path is not yet equivalent to a **feature-complete HiDream-O1-Image runtime**.

The native engine must support the complete publicly exposed HiDream-O1-Image feature surface before performance specialization begins.

The guiding rule is:

> **Do not optimize a subset of the product surface. Freeze and validate every supported execution mode first, then enter M2 with all real sequence shapes, conditioning modes, schedulers, and output paths known.**

---

# 1. Mission

At M1-post completion, the native runtime must expose every functionality publicly supported by the official HiDream-O1-Image family, subject to model-profile capability differences proven by the frozen upstream source.

The required feature families are:

```text
1. Text-to-image generation
2. Native high-resolution generation up to 2048×2048
3. Arbitrary supported aspect ratios / resolution snapping
4. Long-prompt generation
5. Long-text rendering
6. Multilingual text rendering
7. Multi-region text/layout control
8. Instruction-based image editing
9. Single-reference aspect-preserving editing
10. Multi-reference subject-driven personalization
11. Layout-conditioned personalization
12. Skeleton / pose-conditioned personalization
13. Storyboard / sequence-oriented generation
14. Full-model guidance / CFG semantics
15. Dev-model distilled sampling semantics
16. Dev editing scheduler selection
17. Noise-scale / noise-clipping controls
18. Deterministic seed-based native generation
19. Native image input decode/preprocessing
20. Native image output / PNG writing
21. Prompt refinement / Reasoning-Driven Prompt Agent integration
22. Prompt-refiner support for the Dev-2604 profile
23. Progress callback / intermediate preview capability
24. Reusable C API for all modes
25. CLI feature parity with the official inference surface
26. Offline execution of the image model without Python
```

M1-post does not optimize these paths.

It makes them **complete, explicit, reproducible and testable**.

---

# 2. Official feature audit

Before implementation, audit the frozen official upstream and create:

```text
docs/M1_POST_UPSTREAM_FEATURE_AUDIT.md
```

The audit must inspect at least:

```text
README.md
inference.py
models/pipeline.py
models/flash_scheduler.py
model/processor configuration
app.py
prompt_agent.py
prompt_agent_v2.py on the dev branch
technical report where needed
```

Do not implement behavior from third-party forks unless the official source is incomplete and the behavior is separately documented as non-authoritative.

---

# 3. Official capability inventory

The current official source describes HiDream-O1-Image as a Pixel-level Unified Transformer operating directly on raw pixels, text, and task conditions, without an external VAE or a disjoint text encoder.

The officially advertised capabilities include:

```text
text-to-image
long-text rendering
instruction editing
subject-driven personalization
storyboard generation
native high resolution
reasoning-driven prompt refinement
```

The currently exposed official inference examples additionally include:

```text
single-reference editing
multi-reference personalization
multi-reference personalization with skeleton conditioning
multi-reference personalization with layout bounding boxes
Dev editing scheduler selection
reference-aspect preservation
noise scheduling controls
```

M1-post must distinguish:

```text
advertised model capability
```

from:

```text
currently exposed official CLI/API behavior
```

and support both where the frozen upstream provides enough executable semantics.

---

# 4. Important model-profile distinction

The engine currently supports:

```text
Full / Base
Dev
Dev-2604
```

These profiles are not assumed to have identical supported task quality or official status.

## Full / Base

Treat Full as the mandatory complete-feature profile.

It must support all officially exposed image-model tasks:

```text
T2I
single-reference editing
multi-reference personalization
layout conditioning
skeleton conditioning
high-resolution generation
long text / visual text
storyboard-capable composition where upstream semantics exist
```

Default official recipe:

```text
steps = 50
guidance_scale = 5.0
shift = 3.0
scheduler = default
```

unless the frozen source/config says otherwise.

## Dev

The official main branch indicates that Dev can run the core T2I/edit/personalization task family, with different sampling semantics.

Default:

```text
steps = 28
guidance_scale = 0.0
shift = 1.0
scheduler = flash
predefined timesteps
```

For exactly one reference image:

```text
editing scheduler default = flow_match
editing scheduler optional = flash
```

## Dev-2604

The official dev branch describes Dev-2604 as tailored specifically for text-to-image generation and ships a separate Prompt-Refine companion.

Therefore:

```text
DO NOT assume feature parity with Full merely because the transformer topology accepts the input.
```

M1-post must produce a capability matrix from direct oracle validation.

If a mode is not officially valid for Dev-2604:

```text
engine must fail clearly
or
mark it experimental
```

rather than silently claiming support.

---

# 5. Capability matrix

Create:

```text
docs/M1_POST_CAPABILITY_MATRIX.md
```

Minimum columns:

```text
feature
Full official support
Dev official support
Dev-2604 official support
native implementation status
oracle fixture
native fixture
quality/sanity artifact
notes
```

No feature is considered complete without an explicit row.

---

# 6. M1-post.0 — Production runtime contract

## Goal

Replace the validation-harness-centric product surface with a real user-facing runtime while keeping all M1 diagnostic paths intact.

Required native execution contract:

```text
request
    ↓
prompt / references / controls
    ↓
native preprocessing
    ↓
native model inference
    ↓
native output conversion
    ↓
image bytes / file
```

Normal execution must not require:

```text
golden tensors
frozen noise files
Python subprocesses
oracle scripts
test harnesses
network access for the image model
```

---

# 7. Unified request structure

Introduce one model request description capable of representing every mode.

Conceptually:

```c
typedef enum {
    HD_MODE_T2I,
    HD_MODE_EDIT,
    HD_MODE_PERSONALIZE,
    HD_MODE_PERSONALIZE_LAYOUT,
    HD_MODE_PERSONALIZE_SKELETON,
    HD_MODE_STORYBOARD
} hd_mode;

typedef struct {
    const char *prompt;

    hd_mode mode;
    hd_model_profile profile;

    int width;
    int height;
    uint64_t seed;

    int steps;
    float guidance_scale;
    float shift;

    hd_scheduler_kind scheduler;

    const hd_reference_image *references;
    size_t reference_count;

    const hd_layout_condition *layout;
    const hd_skeleton_condition *skeleton;

    bool keep_original_aspect;

    float noise_scale_start;
    float noise_scale_end;
    float noise_clip_std;

    hd_progress_callback progress_cb;
    void *progress_user;
} hd_generation_request;
```

This is illustrative.

Use actual upstream semantics.

Do not create one separate inference engine per task.

---

# 8. Mode inference

Support explicit mode selection.

Automatic mode inference may also be provided:

```text
0 references  -> T2I
1 reference   -> edit
2+ references -> personalization
```

but explicit CLI/API mode must override ambiguity.

Layout/skeleton conditions refine personalization rather than creating unrelated transformer implementations.

---

# 9. M1-post.1 — Native production T2I path

## Required

Implement a user-facing native path:

```text
prompt
seed
width/height
profile
sampling controls
    ↓
PNG
```

No frozen noise fixture required.

No Python.

No network.

---

# 10. Native seed/RNG

Normal user inference must support:

```text
seed -> deterministic native initial state
```

Requirements:

```text
same profile
same prompt
same dimensions
same seed
same engine commit
same precision mode
→ reproducible output
```

Matching PyTorch's RNG bit-for-bit is not required unless separately frozen as a product requirement.

Correctness fixtures may continue to inject explicit noise tensors.

---

# 11. Resolution support

The official interface defaults to:

```text
2048 × 2048
```

and snaps requested dimensions to supported resolutions internally.

The native engine must:

```text
accept width/height
apply the same valid-resolution policy as the frozen oracle
derive image-token count
derive sequence length
derive position/MRoPE metadata
derive workspace
derive final output dimensions
```

No validation-fixture resolution may be hardcoded into production.

---

# 12. Required resolution profiles

At minimum validate:

```text
1024×1024
2048×2048
one portrait aspect
one landscape aspect
```

If the oracle snaps them internally, validate both:

```text
requested dimensions
actual internal dimensions
final returned dimensions
```

---

# 13. No VAE assumption

HiDream-O1-Image is officially described as pixel-level and without an external VAE.

Therefore:

```text
do not add a VAE because conventional diffusion pipelines use one
```

Audit the exact frozen final output path and reproduce it natively:

```text
model/scheduler final state
→ model-specific raw-pixel reconstruction
→ clamp/range conversion
→ RGB image
```

Implement only what the oracle actually does.

---

# 14. M1-post.2 — Long prompts and visual text rendering

Long-text rendering is a first-class advertised capability.

The native runtime must not truncate prompts using assumptions derived from the short M0 canonical prompt.

Validate:

```text
long English prompt
long Chinese prompt
mixed Unicode prompt
quoted visible text
multiple text regions
multi-line requested text
punctuation-heavy text
```

Tokenizer parity remains exact.

---

# 15. Visual-text correctness

This is not a tensor-parity-only feature.

Create a small release sanity corpus including:

```text
single short word
multi-word sign
two separate visible text regions
long English text
Chinese text
mixed-language text
```

The purpose is not benchmark scoring.

It is to catch production-path mistakes such as:

```text
prompt truncation
Unicode corruption
quote stripping
special-token corruption
layout-condition loss
```

---

# 16. Multi-region layout semantics

Do not confuse:

```text
natural-language layout descriptions
```

with:

```text
explicit layout_bboxes conditioning
```

Both must remain supported.

The official CLI accepts layout boxes in multiple forms.

The native parser should support the frozen upstream forms, including:

```text
JSON string
JSON file
array of normalized boxes
object form containing bboxes
box objects carrying optional text where supported
```

Validate exact coordinate ordering from source.

Do not infer x/y ordering from intuition.

---

# 17. M1-post.3 — Single-reference instruction editing

Exactly one reference image selects the editing path.

Required native path:

```text
reference image
+ instruction prompt
+ seed
+ scheduler/profile
→ edited image
```

Validate:

```text
reference decode
reference resize
reference patch/token packing
special-token boundaries
sequence ordering
position IDs
MRoPE
prediction masks
scheduler choice
final output
```

---

# 18. Native reference image decoding

The production runtime must accept common official/example reference formats.

At minimum:

```text
PNG
JPEG
```

Recommended:

```text
WebP if the chosen decoder supports it reliably
```

Audit upstream PIL semantics for:

```text
RGB conversion
alpha handling
EXIF orientation
resize filter
dimension rounding
normalization
```

Parity is based on the actual processor behavior, not assumptions.

---

# 19. Keep-original-aspect behavior

For exactly one reference, support:

```text
--keep-original-aspect
```

The official path resizes the reference with a maximum-size rule and can derive target dimensions from that reference.

Freeze exact semantics:

```text
max size
patch alignment
resolution snapping
final returned dimensions
```

Do not generalize this flag to multi-reference mode unless upstream does.

---

# 20. Dev editing scheduler

For Dev exactly-one-reference editing:

```text
flow_match = default
flash = optional
```

The selector has no effect where upstream says it should be ignored.

Validate both Dev editing schedulers.

For Full, use the Full recipe.

---

# 21. Editing sanity fixtures

Create at least:

```text
color/style edit
object removal
object addition/change
```

One should use:

```text
keep_original_aspect
```

A human sanity artifact is supplementary to tensor/short-run parity.

---

# 22. M1-post.4 — Multi-reference subject personalization

Two or more reference images activate the subject-driven personalization path.

Required:

```text
2 references minimum fixture
3–4 references fixture
many-reference stress fixture
```

The upstream examples demonstrate substantially more than two references.

The engine must not assume a hardcoded count of 2.

---

# 23. Reference count affects preprocessing

Audit and reproduce upstream policies where preprocessing varies with number of references.

For each K:

```text
reference resize policy
max-size rule
token count
sequence layout
reference ordering
position encoding
workspace requirement
```

Do not use the single-reference preprocessing path blindly for K>1.

---

# 24. Reference ordering

Reference ordering must be deterministic and match upstream.

Validate:

```text
ref A, ref B
```

is distinct from:

```text
ref B, ref A
```

where the model semantics depend on order.

No unordered container is allowed in the request representation.

---

# 25. Multi-reference output sanity

Use fixtures that make subject identity observable.

Examples:

```text
same person across new scene
multiple clothing/body-part references
multiple subjects if upstream supports them
```

Do not use only generic landscapes where broken personalization would be invisible.

---

# 26. M1-post.5 — Layout-conditioned personalization

Implement the explicit layout-conditioning path exposed by official inference.

Inputs:

```text
multiple reference images
prompt
layout_bboxes
```

Freeze exact accepted formats and normalization.

Validate:

```text
one bbox per reference where required
multiple bboxes
bbox order
optional text association
out-of-range rejection
malformed JSON rejection
```

---

# 27. Layout data structure

Internally do not pass raw JSON through model code.

Parse once into typed normalized structures.

Conceptually:

```c
typedef struct {
    float x1;
    float x2;
    float y1;
    float y2;
    const char *text;
} hd_layout_box;
```

Use the exact upstream coordinate semantics.

---

# 28. Layout validation

Capture from oracle:

```text
parsed normalized boxes
condition token representation
sequence metadata
position metadata
selected block checkpoint
```

Then validate native.

A final image sanity should visibly test spatial placement.

---

# 29. M1-post.6 — Skeleton / pose-conditioned personalization

The official examples expose skeleton-conditioned personalization through a multi-reference input set including pose/skeleton imagery.

Do not assume the semantics are equivalent to a generic additional reference.

Audit exact source behavior for:

```text
reference-role detection
openpose/skeleton image handling
background reference
face reference
part references
sequence ordering
special tokens / labels
```

If upstream identifies roles through ordering or naming conventions, reproduce that behavior exactly and document it.

---

# 30. Typed reference roles

Do not force the public C API to depend permanently on filename conventions.

After reproducing the official CLI behavior, expose typed internal roles where semantics exist:

```text
subject
face
background
skeleton/openpose
part
generic reference
```

CLI compatibility may map filenames/order into these structures.

---

# 31. Skeleton validation

Required:

```text
oracle preprocessing parity
reference-role/order parity
one-forward checkpoint parity
short generation parity
full sanity image
```

Choose a pose where failure would be visually obvious.

---

# 32. M1-post.7 — Full versus Dev sampling recipes

Do not reduce profile selection to a weight path.

A model profile includes:

```text
weights
steps
guidance behavior
shift
scheduler
timesteps
noise rules
feature capability matrix
```

Freeze all defaults from upstream.

---

# 33. Full / Base CFG

Full uses guidance where the official recipe requires it.

Validate:

```text
guidance_scale default
guidance_scale override
conditional branch
unconditional branch
combination formula
```

Dev guidance behavior differs and must remain profile-driven.

---

# 34. Noise scheduling controls

Expose and implement the official controls:

```text
noise_scale_start
noise_scale_end
noise_clip_std
```

Freeze:

```text
default values
interpolation rule
where noise is injected
clipping formula
which schedulers use these controls
```

Do not ignore arguments silently.

---

# 35. Scheduler matrix

Create a table for:

```text
Full T2I
Full edit
Full personalization
Dev T2I
Dev edit flow_match
Dev edit flash
Dev personalization
Dev-2604 T2I
```

For each record:

```text
scheduler
steps
timesteps source
shift
guidance
noise controls
model-timestep transform
```

This table becomes a test fixture.

---

# 36. M1-post.8 — Storyboard generation

Storyboard generation is an advertised model capability but is not exposed as a distinct command in the current basic official CLI examples.

Therefore M1-post must not invent semantics.

First perform a source/report audit.

Determine whether storyboard behavior is:

```text
A. a dedicated sequence/conditioning mode,
B. multi-reference personalization used across panels,
C. prompt-agent orchestration over repeated generations,
D. another mechanism described in the technical report.
```

Only after this audit implement the correct behavior.

---

# 37. Storyboard support requirement

The engine is considered feature-complete only when it provides an explicit supported storyboard workflow consistent with upstream.

At minimum this must provide:

```text
multiple panels/frames
consistent subject/environment conditioning
deterministic per-panel seed/control
documented reference carry-over semantics
output naming/order
```

If the official implementation is orchestration rather than a distinct model graph, keep it in the orchestration layer rather than adding fake transformer functionality.

---

# 38. Storyboard CLI/API

Do not bake storyboard into the low-level forward.

Preferred model:

```text
storyboard request
→ panel specification
→ one or more ordinary generation requests
→ optional shared references/state
→ ordered outputs
```

Exact behavior follows the source audit.

---

# 39. M1-post.9 — Reasoning-Driven Prompt Agent

The official project includes a prompt reasoning/refinement component.

This is a companion model/service, not the 8B image transformer itself.

Nevertheless it is part of the publicly supported product functionality and must be integrated.

---

# 40. Main-branch Prompt Agent

Official behavior includes:

```text
local backend using Gemma-4-31B-it
OpenAI-compatible API backend
```

Output includes structured:

```text
prompt
reasoning
resolved_knowledge
```

The rewritten `prompt` is then fed to image inference.

M1-post must preserve this interface.

---

# 41. Dev-2604 Prompt Refiner

The dev branch ships a dedicated Prompt-Refine companion and uses an OpenAI-compatible serving interface, typically via a local vLLM endpoint.

Support this as a distinct refiner profile.

Do not assume the Full prompt agent and Dev-2604 refiner are interchangeable.

---

# 42. Prompt Agent runtime boundary

The core image engine must remain independently usable:

```text
raw prompt -> image
```

Prompt refinement is optional.

The image runtime itself still has:

```text
Python dependency = NO
```

The prompt-refiner backend may be:

```text
external OpenAI-compatible service
local separately hosted model service
future native LLM runtime
```

Do not require the image process to embed Python.

---

# 43. Prompt refinement CLI

Support an interface such as:

```text
--refine-prompt
--refiner-backend none|api|local-service
--refiner-base-url
--refiner-model
--refiner-api-key-env
```

Do not put API keys directly into logs or metadata.

Allow saving:

```text
raw_prompt
refined_prompt
resolved_knowledge
```

where privacy policy permits.

---

# 44. Prompt-refiner validation

Validate:

```text
refiner disabled
Full/main prompt agent compatible response
Dev-2604 Prompt-Refine compatible response
non-English raw request
visible-text request
complex spatial request
```

The image engine only needs the final refined prompt.

Reasoning text is not required by the transformer.

---

# 45. M1-post.10 — Progress callbacks and previews

The official web demo exposes per-step progress and optional intermediate previews.

The C API should support at least:

```text
step index
total steps
progress callback
```

Recommended:

```text
optional preview callback
```

without changing model semantics.

---

# 46. Progress callback contract

Callbacks must not:

```text
change scheduler state
change stream ordering
add mandatory D2H every step
slow normal inference when disabled
```

When preview is requested:

```text
produce preview lazily
```

Only selected milestones need preview generation.

---

# 47. Cancellation

The current official web app exposes progress but not a robust cancellation path.

M1-post may add native cancellation if easy, but it is not required for strict upstream parity.

If added:

```text
check cancellation at safe step boundaries
clean resources deterministically
```

Do not interrupt CUDA work unsafely.

---

# 48. M1-post.11 — Native output surface

Support:

```text
PNG file output
in-memory RGB/RGBA buffer
metadata JSON
optional raw tensor output for diagnostics
```

JPEG output is optional.

PNG is required because official inference writes images directly.

---

# 49. Output metadata

Every generated artifact should optionally record:

```text
engine commit
model profile
model revision
prompt
refined prompt if used
reference count
reference hashes/identifiers
mode
width
height
actual internal width/height
steps
seed
scheduler
guidance
shift
noise controls
precision
runtime duration
```

Do not store source reference images unless explicitly requested.

---

# 50. M1-post.12 — CLI parity

At completion, the native CLI must support the semantic equivalent of the official inference arguments.

Required:

```text
--model-path
--model dev|base/full
--prompt
--ref-image (repeatable)
--output
--width
--height
--seed
--steps
--shift
--guidance-scale
--noise-scale-start
--noise-scale-end
--noise-clip-std
--editing-scheduler flow_match|flash
--keep-original-aspect
--layout-bboxes
```

Add explicit:

```text
--mode
```

even if auto-detection is also available.

---

# 51. Recommended extra CLI

Recommended:

```text
--refine-prompt
--refiner-backend
--metadata-output
--progress
--preview-dir
--dry-run
```

`--dry-run` should print:

```text
resolved profile
resolved mode
resolved dimensions
reference count
scheduler recipe
estimated sequence length
workspace requirement
```

without executing model inference.

---

# 52. Error handling

Feature-complete runtime requires mode-aware validation.

Examples:

```text
edit with 0 refs -> fail
edit with >1 ref -> fail
subject mode with <2 refs -> fail
keep-original-aspect with !=1 ref -> fail or ignore exactly as upstream
layout bbox count mismatch -> fail
invalid bbox JSON -> fail
unsupported profile/mode -> fail
unsupported scheduler for profile -> fail
missing reference file -> fail
unsupported resolution -> snap or fail exactly as upstream
```

Do not silently reinterpret invalid requests.

---

# 53. C API feature parity

CLI must be a thin layer over the public C API.

Do not implement functionality only in CLI parsing.

The C API must be able to represent:

```text
all model profiles
all generation modes
all scheduler recipes
all references
layout conditions
skeleton conditions
prompt refinement result
progress callback
output destination
```

---

# 54. Image preprocessing parity

Create a dedicated module:

```text
src/image/
```

or equivalent.

It should handle:

```text
decode
RGB conversion
orientation
resize
crop/pad if upstream uses them
patch alignment
normalization
reference preprocessing
output conversion
PNG encode
```

Do not scatter image processing through transformer code.

---

# 55. Unified sequence builder

Create one request-to-sequence layer that handles:

```text
T2I
edit
multi-ref
layout
skeleton
storyboard panel
```

It should produce explicit:

```text
token IDs
vision/reference pixels or patches
special token positions
prediction mask
position IDs
MRoPE metadata
target image token range
reference token ranges
```

The transformer remains mode-agnostic where upstream permits.

---

# 56. Sequence manifest diagnostics

For every mode, provide a diagnostic command that prints:

```text
text tokens
reference count
reference token counts
target token count
special-token indices
total S
position-ID shape
MRoPE sections
prediction-mask spans
workspace estimate
```

This will be critical before M2 specialization.

---

# 57. Required frozen mode fixtures

Create one canonical fixture per execution mode:

```text
POST_T2I_1024
POST_T2I_2048
POST_T2I_LONG_TEXT
POST_EDIT_1REF
POST_EDIT_1REF_KEEP_ASPECT
POST_SUBJECT_2REF
POST_SUBJECT_MULTIREF
POST_SUBJECT_LAYOUT
POST_SUBJECT_SKELETON
POST_STORYBOARD
```

Prompt-refiner fixtures are separate.

---

# 58. Oracle capture policy

The previous M1 low-run restriction is lifted.

For M1-post:

> **Run as many full oracle/native executions as necessary to establish feature completeness.**

Still required:

```text
oracle SHA recorded
model revision recorded
correct Python env
oracle tree clean
run ledger entry
offline model files
```

Do not avoid a decisive feature test merely to save a run.

---

# 59. Capture enough data this time

For each new mode, one well-designed oracle capture should include enough diagnostic checkpoints to avoid repeated missing-data problems.

Recommended:

```text
preprocessed inputs
sequence manifest
embedding output
every block output if storage is manageable
final norm
final head
raw output
scheduler states at selected steps
final image
```

For new mode bring-up, capturing all block outputs is acceptable.

---

# 60. Correctness levels

Reuse M1 levels:

```text
V0 static/config
V1 preprocessing/startup
V2 local block/module
V3 one whole model forward
V4 1–3 steps
V5 full Dev generation
V6 full Full/Base generation
```

For M1-post feature closure, use the cheapest level during debugging but run a full sanity generation for every user-visible mode.

---

# 61. Mode-level validation ladder

For each mode:

```text
1. preprocessing parity
2. sequence manifest parity
3. one-forward parity
4. 1-step parity
5. 3-step parity
6. full native sanity output
```

Full oracle generation is strongly recommended when the mode changes scheduler/control semantics.

---

# 62. Full/Dev feature testing strategy

Do not multiply every full test unnecessarily.

Mandatory:

## Full

Full must execute all complete model features that upstream supports.

At minimum full sanity for:

```text
T2I
editing
multi-ref personalization
layout personalization
skeleton personalization
storyboard workflow
```

## Dev

Validate all official Dev-supported paths.

At minimum:

```text
T2I full generation
edit flow_match
edit flash
personalization if frozen main-branch oracle confirms support
```

## Dev-2604

Mandatory:

```text
T2I
Prompt-Refine integration
long-text / visible-text sanity
2048 sanity where practical
```

Other modes only if direct frozen-oracle support is proven.

---

# 63. Long-text / multilingual test corpus

Freeze a compact corpus:

```text
English paragraph with quoted sign text
Chinese visible text
mixed English/Chinese
multiple separate labels
long dense descriptive prompt
spatially constrained visible text
```

The goal is feature-path regression, not leaderboard recreation.

---

# 64. Storyboard test corpus

Once storyboard semantics are audited, freeze a minimal 3-panel scenario.

Requirements:

```text
same subject
same environment
three distinct actions/views
deterministic ordering
consistent shared conditions
```

Capture per-panel metadata.

---

# 65. Layout test corpus

Freeze a layout example with two references and two normalized boxes.

Choose visually separated regions.

Validate:

```text
box parsing
box/reference association
condition representation
final spatial placement sanity
```

---

# 66. Skeleton test corpus

Freeze the official-style role set:

```text
face
background
openpose/skeleton
part references
```

if that is what the upstream pipeline actually expects.

Do not rename/reorder semantics until source audit confirms them.

---

# 67. Reference preprocessing stress tests

Test:

```text
portrait JPEG
landscape JPEG
PNG
odd dimensions
alpha PNG
EXIF-oriented JPEG
very large image
small image
```

Expected result must match or intentionally document differences from upstream image decode semantics.

---

# 68. 2048 gate

M1-post must establish whether every relevant mode can run at or up to official 2048 resolution.

For each mode classify:

```text
PASS 2048
PASS but expensive
unsupported by official profile
blocked by reference attention memory
```

If current reference attention prevents official supported 2048 execution, M1-post may declare:

```text
M2 attention optimization is BLOCKING for full 2048 feature closure
```

but this must be explicit.

Do not claim 2048 support while only running 64/1024 fixtures.

---

# 69. M1-post/M2 boundary rule

M2 may start only when one of these is true for every advertised mode:

```text
A. feature works natively and has a sanity artifact
or
B. feature is proven blocked purely by a measured performance/memory limitation whose semantics are already correct and whose resolution is explicitly assigned to M2
or
C. feature is not supported by the selected model profile according to frozen upstream
```

Unknown is not an acceptable status.

---

# 70. Performance specialization freeze

Before M2, freeze all real sequence profiles:

```text
T2I 1024
T2I 2048
edit 1ref
subject 2ref
subject many-ref
layout
skeleton
storyboard
```

Record:

```text
total sequence length
attention shape
workspace
reference-token count
target-token count
scheduler recipe
```

M2 CUDA Graph and shape specialization decisions will use this table.

---

# 71. Progress/preview feature gate

The engine must support:

```text
progress callback disabled -> zero required host output per step
progress callback enabled -> step/total notification
preview requested -> optional image extraction
```

Normal inference must not pay preview cost when disabled.

---

# 72. Web/API compatibility layer

A Flask clone is not required.

However the native API should make it possible to rebuild the official web-demo functionality:

```text
t2i tab
edit tab
subject tab
prompt refinement
progress streaming
intermediate preview
PNG result
```

A small sample HTTP server is optional and should not block M2 unless specifically desired.

---

# 73. Prompt-agent privacy boundary

If an external API refiner is used:

```text
raw prompt leaves the machine
```

Make that explicit.

Offline mode should support:

```text
no refiner
or
local refiner service
```

Do not silently send prompts to a remote endpoint.

---

# 74. Output quality sanity matrix

Create:

```text
artifacts/m1_post/final_sanity/
```

Required outputs:

```text
t2i_1024.png
t2i_2048.png
long_text_en.png
long_text_zh.png
edit_1ref.png
edit_keep_aspect.png
subject_2ref.png
subject_multiref.png
subject_layout.png
subject_skeleton.png
storyboard_00.png
storyboard_01.png
storyboard_02.png
```

Add Dev/Full suffixes where relevant.

---

# 75. Sanity metadata

Each artifact gets JSON:

```text
mode
profile
model revision
engine commit
prompt
reference identifiers/hashes
dimensions
seed
steps
scheduler
guidance
shift
noise controls
runtime
```

Do not embed sensitive external API keys.

---

# 76. Release sanity must be human-reviewable

Tensor parity remains the correctness foundation.

But M1-post exists specifically to close product-surface gaps.

Therefore every user-visible mode must also produce at least one human-inspectable final artifact.

Do not pass:

```text
layout mode
```

using only an internal tensor metric if no layout-conditioned image has ever been generated.

---

# 77. Recommended sub-milestones

```text
M1-post.0   upstream feature audit + capability matrix
M1-post.1   production native T2I + seed + output path
M1-post.2   high-res/aspect + long text / multilingual text
M1-post.3   single-reference editing
M1-post.4   multi-reference personalization
M1-post.5   layout conditioning
M1-post.6   skeleton conditioning
M1-post.7   complete Full/Dev scheduler/control surface
M1-post.8   storyboard/source-defined sequence workflow
M1-post.9   prompt agent / prompt-refiner integration
M1-post.10  progress/preview callbacks
M1-post.11  complete CLI/C API surface
M1-post.12  release sanity matrix + final feature report
```

---

# 78. M1-post.0 gate

Required:

```text
official source audited
all advertised features enumerated
all exposed CLI arguments enumerated
Full/Dev/Dev-2604 capability matrix created
unknown semantics listed as blockers
no implementation assumptions left undocumented
```

Commit:

```text
docs(m1-post): freeze complete hidream feature surface
```

---

# 79. M1-post.1 gate

Required:

```text
normal T2I requires no golden fixtures
native seed path works
1024 generation works
native output conversion works
PNG writing works
offline
no Python
same seed reproducible
M1 numerical suite still green
```

Commits:

```text
feat(m1-post): add native production prompt-to-image path
test(m1-post): validate native t2i release sanity
```

---

# 80. M1-post.2 gate

Required:

```text
2048 model path characterized
portrait/landscape supported
resolution snapping matches oracle
long prompts not truncated incorrectly
Unicode exact at tokenizer layer
English long-text sanity
Chinese long-text sanity
multi-region text sanity
```

Commit:

```text
feat(m1-post): support high-resolution and long-text generation
```

---

# 81. M1-post.3 gate

Required:

```text
single reference decode/preprocess matches oracle
edit sequence matches oracle
Dev flow_match edit passes
Dev flash edit passes
Full edit passes
keep-original-aspect passes
full native edit artifact produced
```

Commit:

```text
feat(m1-post): add native single-reference image editing
```

---

# 82. M1-post.4 gate

Required:

```text
2-reference parity
multi-reference count scaling
reference ordering validated
Full personalization full run
Dev personalization if supported
subject identity sanity artifact
```

Commit:

```text
feat(m1-post): add native multi-reference personalization
```

---

# 83. M1-post.5 gate

Required:

```text
layout JSON forms parsed
bbox semantics frozen
reference-to-box mapping correct
one-forward parity
full layout-conditioned sanity
```

Commit:

```text
feat(m1-post): add reference layout conditioning
```

---

# 84. M1-post.6 gate

Required:

```text
skeleton/reference role semantics frozen
preprocessing parity
sequence parity
one-forward parity
full skeleton-conditioned sanity
```

Commit:

```text
feat(m1-post): add skeleton-conditioned personalization
```

---

# 85. M1-post.7 gate

Required:

```text
Full default recipe exact
Dev T2I recipe exact
Dev edit flow_match exact
Dev edit flash exact
guidance override exact for Full
noise_scale_start exact
noise_scale_end exact
noise_clip_std exact
unsupported combinations fail clearly
```

Commit:

```text
test(m1-post): validate complete scheduler and guidance matrix
```

---

# 86. M1-post.8 gate

Required:

```text
storyboard semantics identified from authoritative source
native/orchestration implementation matches those semantics
3-panel deterministic sanity generated
```

Commit:

```text
feat(m1-post): support storyboard generation workflow
```

If official executable semantics truly cannot be recovered:

```text
STOP
document upstream ambiguity
do not invent compatibility
```

---

# 87. M1-post.9 gate

Required:

```text
prompt refinement optional
main prompt-agent response format supported
Dev-2604 Prompt-Refine response supported
raw/refined prompt boundary explicit
image engine remains Python-free
local-service/offline option documented
```

Commit:

```text
feat(m1-post): integrate hidream prompt refinement backends
```

---

# 88. M1-post.10 gate

Required:

```text
progress callback
step count exact
optional preview generation
zero mandatory preview overhead when disabled
```

Commit:

```text
feat(m1-post): expose generation progress and preview callbacks
```

---

# 89. M1-post.11 gate

Required:

```text
CLI supports complete official surface
public C API supports same semantics
CLI is a thin wrapper
typed references/layout/skeleton
mode validation
clear error paths
```

Commit:

```text
feat(m1-post): expose complete hidream native cli and c api
```

---

# 90. M1-post.12 gate

Required final matrix:

```text
Full T2I
Full edit
Full multi-ref
Full layout
Full skeleton
Full storyboard

Dev T2I
Dev edit flow_match
Dev edit flash
Dev personalization where official

Dev-2604 T2I
Dev-2604 prompt refinement
Dev-2604 other modes only if proven supported

1024
2048 or explicit measured blocker
long English text
Chinese text
portrait/landscape
same-seed determinism
offline image runtime
```

Commit:

```text
test(m1-post): close complete hidream feature parity
```

---

# 91. Documentation outputs

Create:

```text
docs/M1_POST_STATUS.md
docs/M1_POST_REPORT.md
docs/M1_POST_CAPABILITY_MATRIX.md
docs/M1_POST_UPSTREAM_FEATURE_AUDIT.md
docs/M1_POST_MODE_CONTRACTS.md
docs/M1_POST_SCHEDULER_MATRIX.md
docs/M1_POST_CLI.md
```

---

# 92. Mode contracts document

`docs/M1_POST_MODE_CONTRACTS.md` must contain one section per mode:

```text
T2I
Edit
Personalization
Layout
Skeleton
Storyboard
```

Each section includes:

```text
inputs
profile support
preprocessing
sequence layout
scheduler
output
golden fixture
workspace
known limitations
```

---

# 93. M1-post report

`docs/M1_POST_REPORT.md` must include:

```text
feature matrix
oracle revisions
model revisions
native commits
all sanity artifacts
correctness metrics
unsupported combinations
2048 status
prompt-agent status
CLI examples
runtime dependency audit
M2 shape-profile handoff
```

---

# 94. Dependency audit

At closure:

```text
Python required for image inference runtime: NO
network required for image inference runtime: NO
Python required for oracle/testing: YES, offline only
prompt refiner network dependency: OPTIONAL
prompt refiner local service: SUPPORTED where configured
```

Do not conflate optional prompt refinement with the core image engine.

---

# 95. M2 handoff requirements

Before M2 begins, export:

```text
mode
profile
resolution
sequence length
reference token count
target token count
workspace bytes
steps
scheduler
guidance
```

for every real execution profile.

M2 optimization must benchmark the profiles that matter.

Do not tune only T2I if edit/personalization have substantially different shapes.

---

# 96. Do not prematurely optimize during M1-post

Forbidden unless required to make an officially supported feature executable:

```text
custom FlashAttention rewrite
CUDA Graphs
FP8/FP4
kernel fusion
cuBLASLt autotuning
pinned pread loader
arena redesign
shape-specialized kernels
GDS
io_uring
```

Those belong to M2/M3.

Correctness and complete product semantics come first.

---

# 97. Run policy

Unlike early M1:

```text
full oracle runs are allowed whenever useful
full native runs are allowed whenever useful
```

Still:

```text
log every meaningful V2+
freeze fixture identity
never mutate frozen oracle
never silently update revisions
```

The optimization is developer time, not model-run count.

---

# 98. Stop conditions

Stop the affected sub-milestone if:

```text
official semantics are ambiguous
native feature requires a separate duplicated transformer
reference preprocessing cannot be reconciled
profile claims feature support not proven by oracle
layout/skeleton ordering unknown
storyboard semantics cannot be sourced
2048 OOM occurs before memory model is understood
M1 numerical regression appears
```

Do not paper over gaps with visual-only claims.

---

# 99. Recommended execution order

```text
A. Audit official main + dev branch feature surface
B. Build capability matrix
C. Close production T2I path first
D. Close 1024/2048/aspect path
E. Close long-text/multilingual path
F. Implement single-reference editing
G. Implement multi-reference personalization
H. Implement layout conditioning
I. Implement skeleton conditioning
J. Freeze scheduler/profile matrix
K. Audit and implement storyboard workflow
L. Integrate prompt refinement surface
M. Add progress/preview callbacks
N. Complete CLI/C API
O. Generate final sanity matrix
P. Run full M1 regression suite
Q. Write M1_POST_REPORT
R. Mark M1-post COMPLETE
S. Only then start M2
```

---

# 100. Definition of M1-post DONE

```text
[ ] official feature audit complete
[ ] Full/Dev/Dev-2604 capability matrix complete

[ ] native T2I production path
[ ] native deterministic seed path
[ ] native image output path
[ ] PNG output

[ ] 1024 supported
[ ] 2048 supported or explicit measured blocker assigned to M2
[ ] portrait/landscape/aspect handling
[ ] resolution snapping parity

[ ] long prompt support
[ ] English long-text sanity
[ ] Chinese long-text sanity
[ ] multi-region visual-text sanity

[ ] one-reference editing
[ ] keep-original-aspect
[ ] Dev flow_match edit
[ ] Dev flash edit
[ ] Full edit

[ ] 2+ reference personalization
[ ] many-reference preprocessing
[ ] reference ordering parity

[ ] layout conditioning
[ ] layout parsing forms
[ ] layout sanity artifact

[ ] skeleton conditioning
[ ] skeleton role/order parity
[ ] skeleton sanity artifact

[ ] Full scheduler/guidance recipe
[ ] Dev scheduler recipe
[ ] noise controls

[ ] storyboard semantics audited
[ ] storyboard workflow supported
[ ] 3-panel sanity artifact

[ ] main Reasoning-Driven Prompt Agent integration
[ ] Dev-2604 Prompt-Refine integration
[ ] raw prompt mode remains supported

[ ] progress callbacks
[ ] optional preview callbacks

[ ] complete native CLI
[ ] complete public C API
[ ] all invalid combinations fail clearly

[ ] image inference requires no Python
[ ] image inference requires no network
[ ] all M1 correctness gates remain green
[ ] all required release sanity artifacts exist
[ ] M1_POST_REPORT.md complete
[ ] engine repository clean
```

Only then write:

```text
M1-post COMPLETE
M2.0 READY
```

and stop.

---

# 101. Canonical commit sequence

Recommended:

```text
docs(m1-post): freeze complete hidream feature surface

feat(m1-post): add native production prompt-to-image path

feat(m1-post): support high-resolution and long-text generation

feat(m1-post): add native single-reference image editing

feat(m1-post): add native multi-reference personalization

feat(m1-post): add reference layout conditioning

feat(m1-post): add skeleton-conditioned personalization

test(m1-post): validate complete scheduler and guidance matrix

feat(m1-post): support storyboard generation workflow

feat(m1-post): integrate hidream prompt refinement backends

feat(m1-post): expose generation progress and preview callbacks

feat(m1-post): expose complete hidream native cli and c api

test(m1-post): close complete hidream feature parity
```

Use additional focused fix commits when necessary.

Do not combine unrelated feature bring-up into one unreviewable commit.

---

# 102. Source-of-truth notes for the agent

The official project currently documents:

```text
Pixel-level unified raw-pixel architecture
T2I
instruction editing
subject personalization
long-text rendering
storyboard generation
up-to-2048 generation
Reasoning-Driven Prompt Agent
layout-conditioned IP generation
skeleton-conditioned IP generation
Full and Dev sampling recipes
Dev editing flow_match/flash choice
reference aspect preservation
noise-scale / noise-clipping controls
web progress/preview behavior
```

Important caveat:

```text
Dev-2604 is described by its dev-branch README as a T2I-tailored checkpoint.
```

Therefore support claims must be model-profile-specific.

---

# 103. Central architectural principle

The native engine should emerge from M1-post as:

```text
                     hd_runtime
                         │
                         ▼
               request / mode parser
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
      T2I              EDIT           PERSONALIZE
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                           generic      layout     skeleton
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
               unified sequence builder
                         │
                         ▼
               one native transformer
                         │
                         ▼
                scheduler/control path
                         │
                         ▼
                  raw pixel output
                         │
                         ▼
                    PNG / buffer
```

Prompt refinement and storyboard orchestration sit above this graph where appropriate.

There must still be:

```text
ONE transformer implementation
ONE model binding system
ONE scheduler abstraction
ONE image preprocessing layer
ONE public C API
```

Feature parity is achieved through explicit conditioning/configuration, not copied execution paths.

---

# 104. Final principle

M1 established:

```text
the native math can be correct
```

M1-post must establish:

```text
the native product actually implements HiDream-O1-Image
```

Only after both are true should M2 answer:

```text
how fast can we make it on GB10?
```
