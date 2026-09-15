# M0 Report — Reproducible Startup Oracle

## Revisions

- **engine commit:** (M0.3 freeze commit, see below)
- **oracle commit:** `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`
- **Dev model revision:** `b6acc2fe452b3120430620dc4354fa442ee081ea`
- **Base model revision/status:** `0b0901d99f200389e138c61946af1185f5f49a13` — `not_downloaded` (valid explicit profile)

## Environment

- **host:** gx10-da63 (Linux 6.17.0-1008-nvidia)
- **architecture:** x86_64
- **GPU:** NVIDIA GB10
- **compute capability:** 12.1
- **CUDA:** torch compiled against cu130 (torch 2.12.1+cu130)
- **Python:** 3.12.3
- **PyTorch:** 2.12.1+cu130
- **Transformers:** 4.57.1 (pinned in `.venv`)

## Gate results

| Gate | Result | Evidence |
|------|--------|----------|
| M0.0 | PASS | `.gitignore`, `config/dev.json`, `config/base.json`, `scripts/m0/*` (static validation clean) |
| M0.1 | PASS | `config/oracle.lock`, `artifacts/m0/oracle/oracle_identity.txt` |
| M0.2 | PASS | `config/models.lock`, `artifacts/m0/models/` inventory |
| M0.3 | PASS | `config/startup_manifest_dev.json`, `artifacts/m0/startup/freeze_dev.json` |

## Run accounting

- transformer forwards executed during M0: **0**
- denoising steps executed during M0: **0**
- complete image generations during M0: **0**

## Repeatability

Two independent startup-only freezes (run A and run B) were executed on `dev`.
Deterministic structural portions agree exactly:

- oracle SHA, model revision, model config (62 keys), dtype, parameter count (8,804,887,792)
- tensor names/shapes/dtypes/numel (759 tensors) via SHA-256 fingerprint `fb7c7dc5752fe23730766b26eb2fe86e48d990795f53944936fd5f4248ec177c`
- canonical tokenization (prompt `"a red fox sits under a cherry blossom tree"`, 17 token IDs)
- scheduler structural config (28 flow_match timesteps + sigmas)

## Handoff to M1

- **exact next M1 task:** begin native (engine-side) implementation of Dev/Base loading and inference, consuming the frozen startup manifest as ground truth.
- **known compatibility risks:** oracle `ROPE_INIT_FUNCTIONS['default']` requires transformers 4.57.x (system 5.12.1 lacks it); `flash_attn` unavailable (use `FA_VERSION=0` fallback).
- **unresolved Base issue:** Base weights not downloaded (deferred per milestone; profile remains valid).
- **startup manifest:** `config/startup_manifest_dev.json` (versioned) + `artifacts/m0/startup/freeze_dev.json` (full, ignored).
- **oracle/model locks:** `config/oracle.lock`, `config/models.lock`.
- **canonical prompt identifier:** `"a red fox sits under a cherry blossom tree"`.
