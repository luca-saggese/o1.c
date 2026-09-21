# Release Validation — R0 (engine freeze)

Gate R0 freezes the engine before release hardening. This document records the
exact commands, the commit, the model hashes and the observed results.

## 1. Commit and environment

```
commit   ac1f4585792082fc625752f3c684b22bf5f10020
         perf(m2): C4.1 two-pass SDPA for the edit/reference decoder
branch   main
tree     clean (no uncommitted modifications)
```

```
GPU        NVIDIA GB10 (DGX Spark)
driver     580.126.09
CUDA       13.0 (V13.0.88)
cuBLAS     13.1.0.3
cuDNN      9.20
compiler   gcc 13.3.0 / nvcc -arch=sm_121 -O2 -std=c++17
```

Build:

```bash
export LD_LIBRARY_PATH=/home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
make            # -> build/hidream
make server     # -> build/hidream-server
```

## 2. Model artifacts

| artifact | path | size (bytes) | SHA256 |
|---|---|---|---|
| Dev BF16 GGUF | `artifacts/models/hidream-o1-dev-bf16.gguf` | 17 609 840 896 | `359671f96655d4e5ca6ebc38b73b3b7dd9dc7c0b518aa5e6a380c6eb4bd60536` |
| Base safetensors | `models/base/` (8 shards) | — | index `33865f9d84679a850080795107f049c3fac81c78d343322f66a36163b9ab53c9` |
| Dev safetensors | `models/dev/` (8 shards) | — | index `33865f9d84679a850080795107f049c3fac81c78d343322f66a36163b9ab53c9` |

GGUF metadata as parsed by the engine:

```
arch=hidream_o1 profile=dev revision=b6acc2fe452b3120430620dc4354fa442ee081ea
dtype=bf16 layers=36  tensors=759  alignment=256  payload=17609778176 bytes
```

## 3. Test suite

`make test` runs `test_weights`, `test_primitives`, `test_tokenizer`,
`test_block`, `test_full_forward`. It stops at `test_full_forward` because of
the **known pre-existing** `complete_output` failure (see §5). All other
targets were built and run individually.

| test | result |
|---|---|
| `test_weights` | PASS — all weight ingestion assertions passed |
| `test_primitives` | PASS — 18 fixture assertions, 0 failed |
| `test_tokenizer` | PASS — ALL TOKENIZER TESTS PASSED |
| `test_block` | PASS — 7 assertions, 0 failed |
| `test_block_tail` | PASS — final_norm nrmse=0.00286 cos=0.999996; complete output nrmse=0.00459 cos=0.999989 |
| `test_full_forward` | **FAIL (known pre-existing)** — 13 passed, 1 failed |
| `test_gguf` | PASS — 759 tensors, round-trip checked 8 tensors |
| `test_gemm_smoke` | PASS — SMOKE PASS |
| `test_sdpa_block` | PASS — all checks passed |
| `test_sdpa_forward` | PASS — all checks passed |
| `test_vision` | PASS — 9 passed, 0 failed |
| `test_png_roundtrip` | PASS — PNG_ROUNDTRIP_OK |
| `test_image` | PASS — ALL IMAGE TESTS PASSED |
| `test_sequence` | PASS — 4 assertions, 0 failed |
| `test_seq_ref` | PASS — 0 assertions, 0 failed |
| `test_ref_alias` | PASS — 0 assertions, 0 failed |
| `test_decode` | PASS — 6 assertions, 0 failed |
| `test_refiner` | PASS — all tests passed |
| `test_progress` | PASS — all tests passed |
| `test_preview` | PASS — all tests passed |
| `test_model_loader` | PASS — all tests passed |
| `test_engine_preload` | PASS — 2 generations, 1 weight load, 1 forward resolve, 1 vision resolve |
| `test_server` | PASS — SERVER_UNIT_OK |
| `bench-sdpa-twopass` (t2i) | PASS — ACCEPT, 22.63 % faster |
| `bench-sdpa-twopass edit` | PASS — ACCEPT, 19.94 % faster |

Commands:

```bash
make test
make test-gguf test-png test-image test-seq test-seq-ref test-ref-alias \
     test-decode test-refiner test-progress test-preview test-engine test-server
./build/test_gguf artifacts/models/hidream-o1-dev-bf16.gguf models/dev
./build/test_engine_preload artifacts/models/hidream-o1-dev-bf16.gguf 0
```

## 4. Canonical production runs

All runs use the production binary `build/hidream` (no debug timing), the
production GGUF, and the frozen recipe parameters.

| run | command | wall | maxRSS | output SHA256 (prefix) |
|---|---|---|---|---|
| Dev T2I 2048 / 28 | `--model dev --model-dir artifacts/models/hidream-o1-dev-bf16.gguf --mode t2i --width 2048 --height 2048 --steps 28 --seed 42` | 85.76 s | 883 MB | `4e79cd4ab639aa17` |
| Dev edit 2048 / 28 | `--mode edit --ref-image _reference2/HiDream-O1-Image/assets/edit/test.jpg --width 2048 --height 2048 --steps 28 --seed 42` | 198.19 s | 1 265 MB | `21ca624be8e41dbe` |
| Base T2I 2048 / 50 | `--model base --model-dir models/base --mode t2i --width 2048 --height 2048 --steps 50 --seed 42` | 319.63 s | 5 161 MB | `46a9a1453c3815e3` |
| Personalize smoke | `--mode personalize --ref-image … --width 1024 --height 1024 --steps 4 --seed 42` | — | — | `901595df9b9eee05` |

Prompts (verbatim):

```
T2I   : A serene mountain lake at sunrise, photorealistic
edit  : Place the person in a snowy mountain landscape
```

The Dev edit output hash `21ca624be8e41dbe…` matches the accepted Part 4
baseline (`artifacts/m2/edit/final.png`), confirming the C4.1 two-pass SDPA
change is hash-neutral on the production path.

## 5. Known pre-existing numerical exceptions

These are **not** release regressions and their thresholds must not be changed.

| test | metric | value | status |
|---|---|---|---|
| `test_full_forward` / `complete_output` | nrmse | 0.15267 | known pre-existing FAIL |
| `test_full_forward` / `complete_output` | cosine | 0.98882141 | known pre-existing FAIL |
| `test_full_forward` / `complete_output` | max_abs | 1.5 | known pre-existing FAIL |

All other `test_full_forward` assertions pass (13/14), including
`block_0_input`, `embedding`, `target_embedding`, `timestep_conditioning`,
`block_0`, `block_mid`, `block_last`, `final_norm_input`, `final_norm`,
`final_head_input`.

## 6. Server smoke

```bash
./build/hidream-server --model dev \
    --model-dir artifacts/models/hidream-o1-dev-bf16.gguf --port 18099
```

| probe | result |
|---|---|
| `GET /v1/models` | `{"object":"list","data":[{"id":"hidream-o1-image-dev",…}]}` |
| `POST /v1/images/generations` (512×512, b64_json) | HTTP 200, 78.9 s, 16 781 320-byte base64 payload |

## 7. Gate result

```
R0 engine freeze: PASS
  - no release-blocking correctness regression
  - the only failing assertion is the documented pre-existing
    complete_output nrmse=0.15267 / cos=0.98882141
  - all canonical production runs complete and produce valid PNGs
  - server serves /v1/models and /v1/images/generations
```

Proceed to R1.