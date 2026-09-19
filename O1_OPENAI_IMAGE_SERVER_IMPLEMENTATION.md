# o1.c — OpenAI-Compatible Image Server Implementation Plan

**Status:** implementation specification for agent  
**Target:** `o1.c` native C/CUDA runtime on GB10  
**Starting point:** `_reference/ds4-server.c` (ported from antirez/ds4; do **not** reimplement the HTTP server from scratch)  
**Primary API compatibility target:** OpenAI Images API  
**Model lifecycle requirement:** selected HiDream model (`dev` or `base`) is loaded **once at server startup** and stays resident until shutdown.

---

## 0. Executive decision

Implement a new executable:

```text
build/hidream-server
```

by **porting and reducing** `_reference/ds4-server.c`.

Do not create a new web framework, do not introduce a third-party HTTP server, and do not design a proprietary image protocol.

The implementation must preserve the useful ds4 server machinery:

```text
POSIX sockets
blocking client threads
bounded request parsing
growable response buffers
JSON parser/helpers
signal-safe shutdown
worker queue
one resident model worker
CORS option
OpenAI-style error responses
GET /v1/models
request body/time limits
graceful draining on shutdown
```

and replace the ds4-specific LLM/session/KV/tool logic with the existing `o1.c` generation runtime.

The first release must implement the standard OpenAI image endpoints:

```text
GET  /v1/models
POST /v1/images/generations
POST /v1/images/edits
```

The server must expose all relevant `o1.c` generation parameters as standard OpenAI fields when an equivalent exists, and as optional `o1_*` extensions when OpenAI has no equivalent.

Do **not** invent a new endpoint such as `/v1/o1/generate`.

---

# 1. Source of truth and reuse policy

## 1.1 Local reference implementation

The repository contains:

```text
_reference/ds4-server.c
```

This is the implementation source to port.

Start by copying it into the production tree, for example:

```text
src/server/o1_server.c
```

Then strip/replace ds4-specific pieces incrementally.

Do **not** start with an empty `server.c`.

Before editing, make a short source map of `_reference/ds4-server.c` identifying these clusters:

```text
KEEP / PORT:
- process-global shutdown state
- stop_signal_handler()
- buf implementation
- xmalloc/xrealloc/xstrdup helpers
- JSON primitives and bounded recursive skip
- HTTP request reader/parser
- socket timeout configuration
- listen_on()
- response writer
- CORS handling
- client thread lifecycle
- server/client counters
- worker job queue
- worker wakeup / condition variables
- graceful shutdown/drain
- /v1/models skeleton
- OpenAI error response helpers
- test-only compilation hooks where useful

REMOVE / REPLACE:
- ds4_engine/session APIs
- token generation
- KV cache
- tool calling
- reasoning
- Anthropic protocol
- chat/completions parsing
- text completion streaming
- DSML/tool replay
- distributed inference
- tensor parallel server policy
- LLM session batching
```

The upstream ds4 server intentionally uses a simple architecture: a small blocking thread per HTTP connection parses a request and hands inference work to resident model worker state. Preserve that architecture. It fits `o1.c` well because the GB10 generation engine should initially execute one GPU generation at a time.

## 1.2 License requirement

The ds4 reference is MIT licensed.

Because substantial server code is being copied/adapted, preserve the MIT copyright/license notice as required by the license. Add an appropriate notice in the derived server source and/or a repository third-party notices file.

Do not silently copy the source without attribution.

Upstream reference:

```text
https://github.com/antirez/ds4
```

---

# 2. Current o1.c constraint that MUST be fixed first

Today the public generation entry point is approximately:

```c
hd_status hd_generate(
    const hd_generation_request *req,
    const char *model_dir,
    int device_id,
    unsigned char **out_rgb,
    int *out_w,
    int *out_h);
```

This API is suitable for a CLI invocation but is wrong for a persistent server because generation currently owns too much model lifecycle.

The server must **not** do this:

```text
HTTP request
-> hd_generate()
-> parse model index
-> read model/GGUF
-> allocate model weights
-> bind model
-> generate
-> free model
-> HTTP response
```

The required server architecture is:

```text
process startup
    |
    +-> select dev OR base
    +-> resolve model path
    +-> select CUDA device
    +-> load model weights once
    +-> create persistent CUDA/cuBLAS/cuDNN state once
    +-> resolve forward bindings once
    +-> resolve vision bindings once
    +-> initialize tokenizer/model-static state once
    +-> initialize reusable generation runtime once
    |
    v
HTTP listen
    |
    +-> request 1 -> generate using resident model
    +-> request 2 -> generate using same resident model
    +-> request 3 -> generate using same resident model
    ...
    |
shutdown
    |
    +-> drain jobs
    +-> free model/runtime once
```

The server must not start listening until model preload succeeds.

---

# 3. Introduce a resident generation runtime

Refactor the runtime before wiring HTTP.

Use an opaque type, naming can vary, but keep the public surface narrow. Recommended shape:

```c
typedef struct hd_generation_engine hd_generation_engine;

typedef struct {
    const char *profile;      /* "dev" | "base" */
    const char *config_dir;
    const char *model_path;   /* safetensors dir OR .gguf */
    int device_id;

    /* Startup-global adapters only, see LoRA section. */
    const hd_lora_config *lora;
} hd_generation_engine_options;

hd_status hd_generation_engine_open(
    hd_generation_engine **out,
    const hd_generation_engine_options *opt);

hd_status hd_generation_engine_generate(
    hd_generation_engine *engine,
    const hd_generation_request *req,
    unsigned char **out_rgb,
    int *out_w,
    int *out_h);

void hd_generation_engine_close(hd_generation_engine *engine);
```

Keep the existing CLI API as a compatibility wrapper:

```c
hd_status hd_generate(...) {
    hd_generation_engine *e = NULL;
    hd_generation_engine_open(&e, ...);
    hd_generation_engine_generate(e, req, ...);
    hd_generation_engine_close(e);
}
```

The server must call `open()` once, `generate()` many times, and `close()` once.

## 3.1 What belongs in `hd_generation_engine`

Hoist every immutable/model-global object currently created inside the request path where practical:

```text
profile/config
model source identity
GGUF/safetensors metadata required after load
hd_weight_store
persistent CUDA weight arena
persistent cuBLAS/cuBLASLt handles
persistent cuDNN state usable across requests
forward weight bindings
vision tower weight bindings
tokenizer/model-static lookup state
device id
model dimensions/profile recipe
static scheduler/profile defaults
optional startup LoRA-merged weights
```

Also hoist reusable workspace state when shape-safe.

Request-specific state stays request-specific:

```text
prompt
expanded @reference aliases
input images
layout JSON
sequence
seed/RNG request state
scheduler state
target/ref patches
VLM conditioning generated from request images
request output RGB
temporary encoded output
```

A first implementation may still allocate request-dependent activation/workspace buffers per request if extracting them is risky, but **model weights and model initialization must never be repeated per HTTP request**.

## 3.2 Preload gate

Before `listen()`:

1. parse server CLI;
2. load the selected profile;
3. resolve the model path;
4. open resident generation engine;
5. resolve all mandatory bindings;
6. fail if any model component is missing;
7. print loaded model information;
8. only then bind/listen on the TCP socket.

Example startup log:

```text
hidream-server: loading model profile=dev device=0
hidream-server: source=artifacts/models/hidream-o1-dev-bf16.gguf
hidream-server: model loaded in 4.8 s, weights resident
hidream-server: model=hidream-o1-image-dev profile=dev
hidream-server: listening on http://127.0.0.1:8000
```

Never print `listening` before the model is ready.

---

# 4. Server CLI

Implement startup options following the ds4 server style.

Minimum:

```text
--model dev|base
--model-dir PATH
--config-dir DIR
--device N
--host HOST
--port PORT
--cors
--max-body-mb N
--queue-depth N
--api-key KEY               optional
--lora FILE[:MULT]          repeatable, startup-global
-h / --help
```

Recommended defaults:

```text
model:       dev
config-dir:  config
device:      0
host:        127.0.0.1
port:        8000
queue-depth: small bounded value, e.g. 8
cors:        off
auth:        off
```

Example:

```bash
./build/hidream-server \
  --model dev \
  --model-dir artifacts/models/hidream-o1-dev-bf16.gguf \
  --device 0 \
  --host 127.0.0.1 \
  --port 8000
```

Base:

```bash
./build/hidream-server \
  --model base \
  --model-dir artifacts/models/hidream-o1-base-bf16.gguf \
  --device 0
```

## Important model rule

The process loads exactly **one** model profile.

A request is not allowed to switch from `dev` to `base` or vice versa.

The request `model` field is an API compatibility identifier. It selects/validates the already loaded model; it does not trigger model loading.

---

# 5. OpenAI standard to implement

As of the current OpenAI Images API, image generation/editing is represented by dedicated Images endpoints.

Implement:

```text
POST /v1/images/generations
POST /v1/images/edits
```

OpenAI's edit endpoint is also the standard endpoint for reference-image workflows with one or more input images.

Current official documentation:

```text
https://developers.openai.com/api/reference/resources/images
https://developers.openai.com/api/docs/guides/image-generation
```

OpenAI also supports image input and generation in the Responses API, including images supplied by URL, base64 data URL, or Files API file ID. That is useful as a later compatibility layer, but do not make Responses API part of the first implementation milestone.

First make the Images API solid.

---

# 6. Endpoints

## 6.1 `GET /v1/models`

Return only the resident model.

For `--model dev`, canonical API id:

```text
hidream-o1-image-dev
```

For `--model base`, canonical API id:

```text
hidream-o1-image
```

Suggested response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "hidream-o1-image-dev",
      "object": "model",
      "created": 0,
      "owned_by": "local"
    }
  ]
}
```

Accepted aliases may include:

```text
o1-dev
dev
```

or, for base:

```text
o1-base
base
hidream-o1-image
```

But `/v1/models` should return a single canonical loaded model.

If a request names the other unloaded profile, return `model_not_found`; never hot-load it.

---

## 6.2 `POST /v1/images/generations`

Content type:

```text
application/json
```

Standard fields to parse:

```text
model
prompt
n
size
quality
output_format
background
compression
user
```

Minimum successful request:

```json
{
  "model": "hidream-o1-image-dev",
  "prompt": "A cinematic portrait of a woman in a red coat",
  "size": "2048x2048"
}
```

Map to `hd_generation_request`:

```text
prompt         -> req.prompt
no images      -> HD_MODE_T2I
size           -> req.width / req.height after HiDream resolution policy
o1_seed        -> req.seed
o1_steps       -> req.steps
o1_scheduler   -> req.scheduler
o1_guidance_scale -> req.guidance_scale
o1_shift       -> req.shift
...
```

---

## 6.3 `POST /v1/images/edits`

Content type:

```text
multipart/form-data
```

This endpoint must support repeated standard image fields:

```text
image[]
```

Also accept `image` as a single-image compatibility spelling if current OpenAI SDKs send it that way, but normalize internally to an ordered array.

Standard fields:

```text
model
prompt
image[]             one or more
mask                optional standard field
n
size
quality
output_format
background
compression
user
```

OpenAI's standard semantics support one or more input images as references. Preserve upload order.

### o1 mapping

Without an explicit `o1_mode`:

```text
1 image   -> edit
2+ images -> personalize
0 images  -> invalid for /images/edits
```

If `o1_layout_bboxes` is present:

```text
-> personalize_layout
```

If the caller explicitly sends `o1_mode`, validate it against the number of images and use it.

Examples:

```text
o1_mode=edit
o1_mode=personalize
o1_mode=personalize_layout
o1_mode=personalize_skeleton
```

Do not expose storyboard until its server semantics are explicit and tested.

---

# 7. Reference image semantics

Do not create a new mandatory reference protocol.

The standard contract is:

```text
image[0]
image[1]
image[2]
...
```

and the order is preserved.

Map this directly to:

```c
hd_reference_image refs[];
```

with the exact upload order preserved through:

```text
reference parser
-> PATH A ref patches
-> PATH B vision tower
-> sequence
```

No reordering.

## 7.1 Named reference aliases

`o1.c` already supports prompt aliases such as:

```text
@person
@shirt
@ref1
@ref2
```

OpenAI Images API has no standard per-image semantic label field.

Therefore named references must remain **optional vendor extensions**, never a requirement.

For multipart edits accept:

```text
o1_reference_aliases
```

as a JSON array string, for example:

```text
["person","shirt","pose"]
```

It must contain either zero entries or exactly the same number of entries as `image[]`.

Then:

```text
image[0] -> alias person + automatic @ref1
image[1] -> alias shirt  + automatic @ref2
image[2] -> alias pose   + automatic @ref3
```

The aliases only affect existing prompt expansion. They must never change tensor/image ordering.

Example:

```bash
curl http://127.0.0.1:8000/v1/images/edits \
  -F 'model=hidream-o1-image-dev' \
  -F 'prompt=@person wearing @shirt using the pose from @pose' \
  -F 'image[]=@person.jpg' \
  -F 'image[]=@shirt.jpg' \
  -F 'image[]=@pose.jpg' \
  -F 'o1_reference_aliases=["person","shirt","pose"]'
```

If no alias list is provided, `@ref1`, `@ref2`, ... still work if the existing alias frontend permits them.

---

# 8. Standard fields versus `o1_*` extensions

Use a standard OpenAI field whenever there is a real semantic equivalent.

Do not make standard fields mean something unrelated.

Use `o1_*` only for HiDream-specific controls.

## 8.1 Per-request fields

| API field | Type | o1.c mapping |
|---|---:|---|
| `prompt` | string | `hd_generation_request.prompt` |
| `model` | string | validate resident model |
| `n` | int | number of generations in this request |
| `size` | string | output resolution request |
| `output_format` | string | response encoder |
| `user` | string | logging only |
| `o1_mode` | string | `hd_mode` |
| `o1_seed` | uint64 | `seed` |
| `o1_steps` | int | `steps` |
| `o1_scheduler` | string | scheduler |
| `o1_guidance_scale` | float | `guidance_scale` |
| `o1_shift` | float | `shift` |
| `o1_noise_start` | float | `noise_scale_start` |
| `o1_noise_end` | float | `noise_scale_end` |
| `o1_noise_clip` | float | `noise_clip_std` |
| `o1_keep_original_aspect` | bool | existing request flag |
| `o1_layout_bboxes` | JSON/string | existing layout parser |
| `o1_reference_aliases` | JSON array | reference aliases |
| `o1_verbose` | bool | request diagnostics only if safe |

If prompt refinement is present in the current production runtime, expose it under explicit vendor-prefixed fields, for example:

```text
o1_prompt_refine
o1_prompt_refine_model
```

but only if it is already a supported local runtime capability. Do not add an external dependency just because the server exists.

## 8.2 Startup-only fields

These cannot safely vary per request in the first server implementation:

```text
profile dev/base
model-dir / GGUF path
CUDA device
startup LoRA set
```

They are process configuration.

---

# 9. LoRA and preload interaction

Current LoRA application mutates model weights.

That is incompatible with arbitrary per-request LoRA switching on a shared resident model unless the runtime gains a non-mutating adapter execution path.

Therefore the first server version must support LoRA as **startup-global model configuration**:

```bash
./build/hidream-server \
  --model dev \
  --lora adapter.safetensors:0.8
```

The server:

```text
load base model
-> apply startup LoRA once
-> begin listening
```

Do not reload/restore 35+ GB of weights for each request just to support request-local adapters.

If an API request supplies an eventual `o1_lora` field before true request-local adapters exist, return:

```text
400 unsupported_parameter
```

and explain that LoRA is configured at server startup.

This is preferable to violating the model-preload requirement.

Merged GGUF+LoRA is also valid as the startup model.

---

# 10. Resolution policy — critical HiDream requirement

Do not blindly honor arbitrary standard image sizes.

HiDream O1 has validated/trained resolution buckets centered around ~4 MP, and we already established that forced 1024x1024 produces unusable generation while 2048x2048 works.

The official bucket set currently used by the upstream pipeline is:

```text
2048x2048
2304x1728
1728x2304
2560x1440
1440x2560
2496x1664
1664x2496
3104x1312
1312x3104
2304x1792
1792x2304
```

The API must therefore distinguish requested size from effective size.

## 10.1 Default

If `size` is absent or `auto`:

```text
T2I: 2048x2048
single-ref + o1_keep_original_aspect: derive supported patch-aligned dimensions using existing runtime behavior
otherwise: use runtime/default valid bucket
```

## 10.2 Explicit size

Parse standard:

```text
WIDTHxHEIGHT
```

By default snap to the closest supported HiDream bucket using the same policy as production CLI/runtime.

Examples:

```text
1024x1024 -> 2048x2048
2048x2048 -> 2048x2048
wide ratio -> closest wide HiDream bucket
```

Log:

```text
hidream-server: request size 1024x1024 snapped to 2048x2048
```

Return the **actual** size in response metadata.

Optional extension:

```text
o1_exact_size=true
```

means fail with `400 invalid_request_error` instead of snapping.

Do not silently execute unsupported 1024 generation.

## 10.3 Conflict rule

If:

```text
o1_keep_original_aspect=true
```

and the caller also requests a non-auto explicit size, reject the ambiguous request unless current CLI semantics already define a single unambiguous precedence rule.

Fail closed rather than silently choosing one.

---

# 11. Standard fields that o1.c does not currently implement

Compatibility does not mean pretending to support semantics that do not exist.

## `mask`

`/v1/images/edits` has a standard optional `mask` field.

If the current native HiDream runtime has no real mask-conditioned inpainting path:

```text
parse the field
validate that it is an image
return 400 unsupported_parameter
```

Do not ignore it and do not pretend the mask was used.

When true mask conditioning is later implemented, OpenAI semantics say the mask applies to the first input image.

## `background=transparent`

If the runtime only generates RGB:

```text
background=auto      -> accept
background=opaque    -> accept
background=transparent -> 400 unsupported_parameter
```

## `quality`

Do not invent an arbitrary quality mapping.

For the first implementation:

```text
quality absent/auto -> profile defaults
```

If the project later defines tested quality presets, map them explicitly and document them.

Until then, unsupported `low/medium/high/...` may return `400 unsupported_parameter` rather than silently altering unrelated knobs.

## `output_format`

Current `o1.c` has a native PNG output path.

Minimum required:

```text
png -> supported
```

If JPEG/WebP encoders are added:

```text
jpeg
webp
```

may be enabled.

Until then, return a clear unsupported-format error. Never return PNG while claiming JPEG.

## `compression`

Only meaningful for JPEG/WebP.

Reject or ignore only according to a documented rule. Prefer rejection when output format cannot use it.

---

# 12. `n` semantics

OpenAI image generation supports generating multiple images with `n`.

Support:

```text
n >= 1
```

with a conservative server limit, e.g.:

```text
1 <= n <= 4
```

The GPU generations may run serially inside one job.

If `o1_seed` is omitted, generate independent seeds using a secure/random source.

If `o1_seed=S` is supplied, define deterministic multi-image behavior:

```text
image 0 -> S
image 1 -> S+1
image 2 -> S+2
...
```

Document this as an `o1_*` deterministic extension behavior.

Do not run N simultaneous generations against a non-reentrant resident engine.

---

# 13. HTTP architecture to port from ds4

Preserve ds4's fundamental split:

```text
client connection thread
    |
    +-> parse/validate HTTP
    +-> parse JSON or multipart
    +-> construct o1 job
    +-> enqueue
    +-> wait for completion
    +-> serialize response
    +-> send response

single model worker
    |
    +-> dequeue
    +-> hd_generation_engine_generate(resident_engine, request)
    +-> encode output
    +-> signal client
```

This is the correct v1 architecture for a single GB10 model.

## 13.1 Why one inference worker

The model runtime owns:

```text
CUDA device
persistent cuBLAS state
cuDNN state
weight arena
large generation workspaces
```

Until explicit reentrancy/multi-request batching exists, exactly one worker may call the generation engine at a time.

HTTP connections may be concurrent. GPU inference is serialized through the job queue.

Do not guard `hd_generate` with ad-hoc mutexes in client threads. Preserve a proper worker queue.

## 13.2 Bounded queue

Add a queue limit.

If full, reply with a standard error such as HTTP 429 or 503; choose one policy and test it.

Example body:

```json
{
  "error": {
    "message": "Server generation queue is full",
    "type": "server_error",
    "param": null,
    "code": "queue_full"
  }
}
```

No unbounded linked-list of 2048-image jobs.

---

# 14. HTTP body and image limits

Images are much larger than ds4 chat requests.

Keep hard limits.

Recommended first pass:

```text
max request body: configurable, conservative
max individual image: 50 MiB
max reference count: HD_SEQ_MAX_REFS
max prompt bytes: bounded
max n: small
```

Do not allocate based on an unchecked `Content-Length`.

The multipart parser must reject:

```text
invalid boundary
truncated body
duplicate singleton fields with conflicting values
too many images
image larger than limit
body larger than limit
missing prompt
invalid model
invalid numeric parameters
```

Keep ds4 socket read/write timeouts.

---

# 15. Multipart parser

`ds4-server.c` is JSON-oriented for its LLM APIs, so multipart handling is new work, but the HTTP transport itself must still be ported rather than rewritten.

Implement a small bounded `multipart/form-data` parser sufficient for OpenAI Images edits.

Required fields:

```text
text fields:
  model
  prompt
  n
  size
  quality
  output_format
  background
  compression
  user
  o1_*

file fields:
  image
  image[]
  mask
```

Preserve the order in which repeated `image[]` parts appear.

Never trust the uploaded filename as a filesystem path.

---

# 16. Input image materialization

The current `hd_reference_image` ABI uses:

```c
const char *path;
```

For the first server implementation, avoid destabilizing the validated image pipeline.

A safe bridge is:

```text
multipart bytes
-> secure temporary file created by server
-> hd_reference_image.path
-> existing native image decoder
-> cleanup after request
```

Requirements:

```text
mkstemp-style unguessable names
0600 permissions
server-owned temp directory
never use raw client filename as path
cleanup success/error/cancel paths
no path traversal
no request-provided server-side local paths
```

Prefer a configurable temp directory and `/tmp`/tmpfs on Linux.

Later, after server parity is frozen, add `hd_image_load_memory()` and remove temporary files if desired.

Do not make image-memory decode a prerequisite for the first working server.

---

# 17. Output image encoding

Do not write the generated response to a user-selected server-side filename.

The API response should return encoded image bytes.

OpenAI's modern image APIs return image data as base64 (`b64_json`).

Add an in-memory PNG encoding path if the current PNG wrapper only writes files:

```c
hd_status hd_png_encode_rgb(
    const uint8_t *rgb,
    int width,
    int height,
    uint8_t **out,
    size_t *out_len);
```

Then:

```text
RGB output
-> PNG bytes in memory
-> base64
-> JSON b64_json
```

Avoid:

```text
RGB -> output.png -> reopen output.png -> base64
```

for the normal server path.

A temporary output file is acceptable only as a very short bring-up step; replace it before the final commit.

---

# 18. Standard response shape

Successful generation/edit response should follow the OpenAI Images shape.

Example:

```json
{
  "created": 1789800000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAA...",
      "output_format": "png",
      "size": "2048x2048",
      "quality": "auto"
    }
  ]
}
```

Do not return:

```text
/tmp/foo.png
file:///tmp/foo.png
local filesystem path
custom binary body
```

as the main API result.

## 18.1 Usage

OpenAI image responses can include token usage fields.

`o1.c` does not need to fabricate OpenAI token accounting.

In v1:

```text
omit usage if there is no trustworthy equivalent
```

Do not return fake token counts.

Optional vendor metadata may be included only if it does not break standard clients, for example:

```json
"o1": {
  "seed": 123,
  "scheduler": "flash",
  "steps": 28,
  "requested_size": "1024x1024",
  "effective_size": "2048x2048"
}
```

Keep the main `data[].b64_json` standard.

---

# 19. Legacy compatibility

Modern GPT Image APIs use base64 image data.

For compatibility with older OpenAI-style clients, optionally accept:

```text
response_format=b64_json
```

as a no-op alias.

Do not implement `response_format=url` unless there is a real HTTP object/file serving subsystem.

Returning a local path as a fake URL is not acceptable.

---

# 20. OpenAI error model

All API failures should use one consistent JSON form:

```json
{
  "error": {
    "message": "Requested model is not loaded",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

Examples:

```text
400 invalid_request_error
400 unsupported_parameter
404 model_not_found
413 request_too_large
415 unsupported_media_type
429 queue_full
500 server_error
```

Specific error codes:

```text
invalid_image
too_many_images
invalid_size
unsupported_size
unsupported_output_format
unsupported_background
mask_not_supported
reference_alias_count_mismatch
invalid_scheduler
invalid_mode
```

Do not leak internal file paths, CUDA pointers, or stack/debug dumps to the API response.

Log internal diagnostics to stderr.

---

# 21. Authentication

The original ds4 server is local-first and recommends TLS/auth in front of it.

Preserve that model by default:

```text
127.0.0.1 only
no required key
```

Add optional:

```text
--api-key SECRET
```

When configured, require:

```http
Authorization: Bearer SECRET
```

Use constant-time comparison if easy.

Without `--api-key`, accept standard Authorization headers but ignore them.

Do not add TLS to this implementation. Document reverse proxy/TLS for remote exposure.

---

# 22. CORS

Port ds4's `--cors` behavior.

When enabled, return suitable:

```text
Access-Control-Allow-Origin
Access-Control-Allow-Headers
Access-Control-Allow-Methods
```

and handle `OPTIONS`.

CORS does not mean authentication.

---

# 23. Operational health endpoint

It is acceptable to add a small non-OpenAI operational endpoint:

```text
GET /healthz
```

Response:

```json
{
  "status": "ok",
  "model": "hidream-o1-image-dev",
  "profile": "dev",
  "ready": true
}
```

This is operational metadata, not a replacement for an OpenAI API endpoint.

It must return ready only after model preload succeeds.

---

# 24. Model names and request validation

The loaded process determines valid model names.

Example dev aliases:

```text
hidream-o1-image-dev
o1-dev
dev
```

Base aliases:

```text
hidream-o1-image
o1-base
base
```

Do not accept a dev request on a base process and silently use base.

Return a model mismatch error.

---

# 25. Standard OpenAI Python client smoke test

The point of compatibility is to work with existing clients.

Add an integration smoke script such as:

```text
tests/server/openai_client_smoke.py
```

Generation:

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

r = client.images.generate(
    model="hidream-o1-image-dev",
    prompt="A cinematic photograph of a red fox in snow",
    size="2048x2048",
    extra_body={
        "o1_seed": 42,
        "o1_steps": 28,
    },
)

open("/tmp/o1-server.png", "wb").write(
    base64.b64decode(r.data[0].b64_json)
)
```

Edit/reference smoke:

```python
with open("example_assets/edit/test.jpg", "rb") as f:
    r = client.images.edit(
        model="hidream-o1-image-dev",
        image=f,
        prompt="remove the earphones",
        extra_body={
            "o1_seed": 42,
            "o1_keep_original_aspect": True,
        },
    )
```

Also test repeated images using the SDK form supported by the installed current OpenAI client.

Do not claim OpenAI compatibility until the official client can complete these calls without custom HTTP code.

---

# 26. Curl examples

## T2I

```bash
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"hidream-o1-image-dev",
    "prompt":"A red fox walking through snow",
    "size":"2048x2048",
    "output_format":"png",
    "o1_seed":42,
    "o1_steps":28
  }' \
  | jq -r '.data[0].b64_json' \
  | base64 --decode > /tmp/fox.png
```

## Single-reference edit

```bash
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F 'model=hidream-o1-image-dev' \
  -F 'prompt=remove the earphones' \
  -F 'image[]=@example_assets/edit/test.jpg' \
  -F 'o1_seed=42' \
  -F 'o1_keep_original_aspect=true' \
  | jq -r '.data[0].b64_json' \
  | base64 --decode > /tmp/edit.png
```

## Multi-reference personalization

```bash
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F 'model=hidream-o1-image-dev' \
  -F 'prompt=@person wearing @shirt' \
  -F 'image[]=@person.jpg' \
  -F 'image[]=@shirt.jpg' \
  -F 'o1_reference_aliases=["person","shirt"]' \
  -F 'o1_mode=personalize' \
  -F 'size=2048x2048' \
  -F 'o1_seed=42' \
  | jq -r '.data[0].b64_json' \
  | base64 --decode > /tmp/personalize.png
```

## Layout

```bash
curl -s http://127.0.0.1:8000/v1/images/edits \
  -F 'model=hidream-o1-image-dev' \
  -F 'prompt=@person and @object arranged according to the supplied layout' \
  -F 'image[]=@example_assets/IP_layout/0.jpg' \
  -F 'image[]=@example_assets/IP_layout/1.jpg' \
  -F 'o1_reference_aliases=["person","object"]' \
  -F 'o1_mode=personalize_layout' \
  -F 'o1_layout_bboxes=[[0.20507812,0.43945312,0.48828125,0.7421875],[0.57617188,0.80078125,0.08789062,0.34179688]]' \
  | jq -r '.data[0].b64_json' \
  | base64 --decode > /tmp/layout.png
```

---

# 27. Job ownership and cleanup

Each queued job owns all request-lifetime resources:

```text
copied prompt
parsed parameters
temporary uploaded image files
reference alias strings
layout string/parsed layout
output RGB
encoded image bytes
error string/status
condition variable / completion state
```

The client thread must not free request memory while the worker is generating.

Recommended lifecycle:

```text
client parses request
-> alloc job
-> enqueue
-> wait job->done
-> send response
-> free response/job/temp images
```

On every failure path:

```text
close fd
free body
free JSON/multipart fields
unlink temporary images
free RGB/output bytes
release queue/job state
```

Port ds4's disciplined ownership patterns.

---

# 28. Shutdown

Keep ds4's signal behavior:

```text
SIGINT/SIGTERM
-> set stop flag
-> close listen fd
-> stop accepting
-> drain or terminate queued work according to one documented policy
-> join worker
-> wait client threads
-> close resident generation engine
-> exit
```

The model must be freed exactly once.

Second SIGINT may hard-exit as ds4 does.

No detached model worker surviving engine teardown.

---

# 29. Progress callbacks

Do not stream CLI progress bars into HTTP responses.

The existing `hd_progress_callback` can be used internally to:

```text
track current step
update server logs
later support image streaming/progress
```

For v1, normal OpenAI Images response is returned only at completion.

Do not implement SSE/partial images in the first milestone.

OpenAI does have image partial streaming, but it is separate work.

---

# 30. Files API and Responses API — later phases

The current OpenAI multimodal ecosystem also supports:

```text
POST /v1/files
POST /v1/responses
```

and Responses can accept input images as:

```text
fully qualified URL
base64 data URL
file_id
```

This is useful eventually, but do not delay the first image server for it.

Roadmap:

```text
Phase 1:
  /v1/models
  /v1/images/generations
  /v1/images/edits

Phase 2:
  /v1/files
  file_id image storage/reuse

Phase 3:
  /v1/responses
  input_image URL/data URL/file_id
  image_generation output/tool compatibility
```

Do not add fake partial Responses support just to advertise the endpoint.

---

# 31. Build layout

Recommended source tree:

```text
src/server/
    o1_server.c
```

Keep the first port mostly monolithic because `_reference/ds4-server.c` is already a tested monolithic server and unnecessary splitting would make the port harder to review.

After correctness, helpers may be extracted if there is clear value.

Add:

```text
make server
```

or build it with default `make`, producing:

```text
build/hidream-server
```

Link against the same CUDA/cuBLAS/cuDNN/image/runtime objects as `build/hidream`.

Do not compile a second copy of inference mathematics.

---

# 32. Server tests

## 32.1 Model-free tests

Port the ds4 test style where useful and test:

```text
JSON parser
HTTP request line/headers
Content-Length bounds
multipart boundary parser
repeated image[] ordering
field normalization
OpenAI error JSON
model alias validation
size parser/snap policy
o1_* parsers
reference alias list
queue overflow
CORS/OPTIONS
```

These must not need model weights.

## 32.2 Preload lifecycle test

Start server with timing/debug counters.

Send two requests.

Require:

```text
model open/load count == 1
weight load count == 1
forward binding resolve count == 1
vision binding resolve count == 1
requests completed == 2
model close count == 1 at shutdown
```

There must be no model file reads caused by the second request.

This is a release gate.

## 32.3 CLI/server equivalence

For a fixed request:

```text
profile
prompt
seed
size
steps
scheduler
guidance
shift
noise params
refs
```

generate once via CLI and once via server.

The server must use the **same engine/runtime path**.

For deterministic paths, require identical decoded RGB/PNG content or the same numerical/image contract already used by the project.

At minimum freeze:

```text
T2I Dev 2048
single-ref edit Dev 2048 / keep-aspect equivalent
multi-ref personalization
Base T2I
```

## 32.4 Concurrency/queue test

Launch two or more clients concurrently.

Require:

```text
both HTTP connections accepted
jobs queued
only one generation engine call active at a time
results delivered to correct clients
no request state crossover
```

Seed/reference state from job A must never appear in job B.

## 32.5 Failure tests

Test:

```text
bad JSON
bad multipart
missing prompt
no image on edits
too many refs
invalid image
request too large
unknown model
unsupported mask
unsupported transparent background
unsupported output format
invalid o1_mode
invalid layout JSON
alias count mismatch
queue full
client disconnect
SIGINT during idle
SIGINT while request active
```

---

# 33. README documentation

Add a "Server / OpenAI-compatible Images API" section with:

```text
startup commands
resident model behavior
supported endpoints
supported standard fields
supported o1_* fields
known unsupported fields
curl examples
OpenAI Python SDK example
resolution snapping warning
security warning
```

Explicitly document:

> The server loads one Dev or Base model at startup and does not switch model weights per request.

And:

> HiDream O1 generation is validated at its native high-resolution buckets. A request such as `1024x1024` may be snapped to `2048x2048`; inspect the returned effective `size` or use `o1_exact_size=true` to reject snapping.

---

# 34. Security constraints

The server handles arbitrary binary uploads, so hardening is part of correctness.

Mandatory:

```text
body size limit
per-image size limit
bounded JSON nesting (retain ds4 protection)
bounded multipart part count
bounded header size
socket IO timeout
no uploaded filename path usage
no server-side path supplied by API client
secure temp filenames
cleanup temp files
no shell execution
no command-line construction from request strings
no external URL fetching in v1
```

The server calls C APIs directly.

Never implement generation by constructing:

```text
system("./build/hidream ... user prompt ...")
```

No shell.

---

# 35. Performance constraints

The whole reason for a resident server is to remove startup cost.

Request critical path must **not** include:

```text
GGUF/safetensors model loading
model weight H2D load
cuBLAS handle creation
cuDNN global initialization
forward binding resolution
vision weight binding resolution
profile parsing
```

Request path may include:

```text
prompt tokenization
reference image decode/preprocess
request sequence construction
request workspace preparation
denoise
RGB reconstruction
PNG encoding/base64
```

Log or measure:

```text
queue_wait_ms
request_parse_ms
preprocess_ms
generation_ms
encode_ms
total_ms
```

Diagnostic timing must not add CUDA synchronizations to production inference.

---

# 36. Do not break the existing CLI

The server refactor must leave:

```text
build/hidream
```

working.

The CLI and server must share the same resident-engine API internally.

Desired architecture:

```text
                    +------------------+
CLI ----------------> hd_generation_   |
                    | engine           |
HTTP worker --------> generate()       |
                    +------------------+
                         |
                         v
                  one production
                  inference path
```

Not:

```text
CLI -> hd_generate implementation A
server -> copied generation implementation B
```

No duplicate inference logic.

---

# 37. Implementation order for the agent

Execute in this order and do not branch into unrelated optimization work.

## Commit 1 — resident generation engine

1. introduce resident engine abstraction;
2. move model load/bind state out of one-shot request lifetime;
3. keep old CLI working through compatibility wrapper;
4. prove two generations can run sequentially against one preload;
5. tests.

Suggested commit:

```text
refactor(runtime): add resident generation engine
```

## Commit 2 — port ds4 HTTP core

1. copy `_reference/ds4-server.c`;
2. preserve attribution;
3. remove ds4 model/tool/KV/chat code;
4. retain HTTP/socket/thread/queue/error/CORS machinery;
5. create `build/hidream-server`;
6. implement `/healthz` and `/v1/models`;
7. start listening only after resident model init.

Suggested commit:

```text
feat(server): port ds4 resident HTTP server core
```

## Commit 3 — image generations endpoint

1. `/v1/images/generations`;
2. JSON parser fields;
3. `o1_*` mapping;
4. resolution snapping;
5. in-memory PNG/base64 response;
6. CLI/server equivalence test.

Suggested commit:

```text
feat(server): add OpenAI image generations API
```

## Commit 4 — edits and references

1. multipart parser;
2. repeated `image[]`;
3. secure upload temp files;
4. mode inference;
5. reference aliases;
6. layout options;
7. single and multi-reference tests;
8. unsupported mask behavior explicit.

Suggested commit:

```text
feat(server): add OpenAI image edits and references
```

## Commit 5 — closeout

1. OpenAI Python SDK smoke;
2. concurrent queue test;
3. cleanup/error paths;
4. README;
5. remove debug;
6. full clean build/tests;
7. git clean.

Suggested commit:

```text
test(server): close OpenAI image API compatibility
```

---

# 38. Definition of done

Do not declare the task complete until all of these are true:

```text
[ ] Server implementation is derived from _reference/ds4-server.c, not rewritten from zero.
[ ] ds4 MIT attribution/license requirement is preserved.
[ ] build/hidream-server exists.
[ ] --model dev|base selects exactly one startup-resident model.
[ ] Model/GGUF/safetensors weights load before listen().
[ ] Model weights load exactly once per process.
[ ] Two sequential HTTP generations do not reload model weights.
[ ] GET /v1/models returns loaded model.
[ ] POST /v1/images/generations works.
[ ] POST /v1/images/edits works for one reference.
[ ] POST /v1/images/edits works for multiple references.
[ ] image[] order is preserved.
[ ] Optional o1_reference_aliases maps cleanly to existing @alias frontend.
[ ] o1 layout/reference controls are wired.
[ ] All ordinary generation knobs are represented by standard or o1_* fields.
[ ] Startup-only model/device/LoRA configuration is documented.
[ ] Unsupported standard mask behavior fails explicitly, not silently.
[ ] Unsupported transparent background fails explicitly.
[ ] Resolution policy prevents accidental bad 1024 HiDream runs.
[ ] Output is standard data[].b64_json.
[ ] No server-side output path is required from API caller.
[ ] Input uploads are bounded and securely materialized.
[ ] Inference runs through a bounded worker queue.
[ ] Only one non-reentrant engine generation is active at a time.
[ ] Concurrent HTTP clients receive the correct independent results.
[ ] CLI and server share the same production generation implementation.
[ ] OpenAI Python SDK generation smoke passes.
[ ] OpenAI Python SDK edit smoke passes.
[ ] SIGINT/SIGTERM drains/cleans correctly.
[ ] make clean && make && server tests pass.
[ ] No debug prints/temp artifacts remain.
[ ] README is updated.
[ ] git status is clean.
```

---

# 39. Explicit non-goals for this milestone

Do not spend this milestone on:

```text
Responses API
Files API
remote image URLs
SSE partial image streaming
GPU request batching
multiple simultaneously resident Dev+Base models
per-request mutable LoRA swapping
TLS
distributed inference
new image-generation algorithms
new scheduler math
new CUDA optimization
```

Those are follow-up tasks.

The purpose of this milestone is:

> **port the proven ds4 HTTP/worker architecture, keep one HiDream model resident, and expose the existing validated o1.c generation engine through the standard OpenAI Images API.**

---

# 40. Final architectural picture

```text
                 startup
                    |
                    v
        +-------------------------+
        | parse server CLI        |
        | --model dev|base        |
        | --model-dir             |
        | --device                |
        +-------------------------+
                    |
                    v
        +-------------------------+
        | hd_generation_engine    |
        | OPEN ONCE               |
        |                         |
        | weights resident        |
        | bindings resident       |
        | cuBLAS/cuDNN resident   |
        +-------------------------+
                    |
                    v
        +-------------------------+
        | ds4-derived HTTP core   |
        | listen 127.0.0.1:8000   |
        +-------------------------+
              /           \
             /             \
            v               v
   client thread A     client thread B
       parse              parse
         |                  |
         +------ queue ------+
                    |
                    v
          +------------------+
          | generation worker|
          | serialized GPU   |
          +------------------+
                    |
                    v
          hd_generation_engine_generate()
                    |
                    v
             RGB output
                    |
                    v
          PNG memory encode
                    |
                    v
              base64
                    |
                    v
         OpenAI Images JSON
         data[].b64_json
```

This is the architecture to freeze before adding later API surfaces.

---

# References

OpenAI current image API/reference:

- https://developers.openai.com/api/reference/resources/images
- https://developers.openai.com/api/docs/guides/image-generation

Reference server:

- local: `_reference/ds4-server.c`
- upstream: https://github.com/antirez/ds4
- upstream server documentation: https://github.com/antirez/ds4/blob/master/docs/SERVER.md

Relevant current `o1.c` components to inspect/refactor:

```text
src/runtime/generate.c
src/runtime/generate.h
src/runtime/request.c
src/runtime/request.h
src/model/weights.c
src/model/weights.h
src/model/forward.c
src/model/forward.h
src/model/vision.c
src/model/vision.h
src/main.c
src/image/
include/hidream.h
Makefile
```
