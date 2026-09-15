# M0 Execution Status

<!-- Maintain this file continuously. It must always show the fields below. -->

- **Current M0 sub-step:** M0.3 (freeze startup oracle) — COMPLETE
- **Last completed gate:** M0.3 (startup freeze + repeatability + offline)
- **Current engine Git commit:** `e3888b7` (M0.2 pin; M0.3 commit pending)

- **Upstream oracle SHA:** `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`
- **Dev HF revision:** `b6acc2fe452b3120430620dc4354fa442ee081ea`
- **Base HF revision/path status:** `0b0901d99f200389e138c61946af1185f5f49a13` — `not_downloaded` (explicit profile, valid config)

## Gate status

| Gate | Status | Evidence |
|------|--------|----------|
| M0.0 scaffold + low-run policy | PASS | `.gitignore`, `config/dev.json`, `config/base.json`, `scripts/m0/*`, `bash -n` + `py_compile` clean |
| M0.1 pin official oracle | PASS | `config/oracle.lock`, `artifacts/m0/oracle/oracle_identity.txt`, imports succeed |
| M0.2 pin Dev weights | PASS | `config/models.lock`, local-only resolution OK, 24 files / 33 GB |
| M0.3 freeze startup oracle | PASS | `config/startup_manifest_dev.json`, `artifacts/m0/startup/freeze_dev.json`, 2 freezes agree |

## Commands executed

- `git status --short`, `git log --oneline` — repository survey.
- `git ls-remote https://github.com/HiDream-ai/HiDream-O1-Image.git` → `dev` = `3237a638...`.
- `./scripts/m0/01_checkout_oracle.sh` — cloned/pinned oracle at `/python` (detached, clean).
- `.venv` (system-site-packages) + `transformers==4.57.1 diffusers accelerate einops`.
- `FA_VERSION=0` import validation → all oracle modules import cleanly.
- `./scripts/m0/02_download_models.sh dev` — 24 files / 33 GB into `models/dev`.
- `./scripts/m0/03_freeze_startup.sh dev` — run 4x total; final two runs (A/B) structurally agree (tensor fingerprint `fb7c7dc5…`, config, tokenizer IDs, scheduler all equal).

## Commands failed

- `python3` (system transformers 5.12.1) KeyError `'default'` in `ROPE_INIT_FUNCTIONS` — fixed by pinning freezer to `.venv/bin/python` (transformers 4.57.1).
- (historical) `flash_attn` import under default `FA_VERSION=2` — fixed with `FA_VERSION=0`.

## Unresolved issues

- (none blocking M0)

## Next action

Commit M0.3 freeze (`test(m0): freeze reproducible python startup oracle`), finalize `docs/M0_REPORT.md`, verify clean repos, report completion and STOP (do not start M1).
