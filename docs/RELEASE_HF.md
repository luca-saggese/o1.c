# R4 — Hugging Face release repository

**Gate status: READY TO UPLOAD — target resolved, artifacts verified, upload
requires credentials that are absent in this environment.**

Per the release-hardening rules, this gate does **not** invent success. All
release materials are staged and verified locally; publication requires a
Hugging Face token that is not present here.

## 1. Target repository — RESOLVED

| Field | Value |
|-------|-------|
| Repository | `saggeseluca/o1.c-models` |
| Type | model |
| Visibility | public |
| Gated | no |
| URL | <https://huggingface.co/saggeseluca/o1.c-models> |
| Current contents | `.gitattributes` only (empty repo) |

Verified:

```text
GET https://huggingface.co/api/models/saggeseluca/o1.c-models  -> 200
```

This is the namespace referenced by `hf_release_repo` in
[`models/manifest.json`](../models/manifest.json), by the `download_url` of
every entry, by [`scripts/download_model.sh`](../scripts/download_model.sh)
and by the README. The manifest is the single source of truth; no URL is
hard-coded in a second place.

## 2. Credential check

| Check | Result |
|-------|--------|
| `HF_TOKEN` in environment | absent |
| `~/.cache/huggingface/token` | absent |
| `hf auth whoami` | `Error: Not logged in` |
| `hf` CLI available | yes, version 1.19.0 |

Conclusion: **cannot authenticate**, therefore **cannot upload**. The upload
script detects this and exits with code 2 before touching the network.

## 3. Upload procedure (maintainer)

Two commands. The uploader takes the target from the manifest, so no argument
is needed.

```bash
# 1. authenticate (interactive, or: export HF_TOKEN=hf_xxx)
hf auth login

# 2. dry-run, then upload
./release/hf_upload/upload.sh --dry-run
./release/hf_upload/upload.sh
```

Expected payload (7 files, ~52.8 GB total):

```text
hidream-o1-dev-2604-bf16.gguf   16.4 GiB
hidream-o1-dev-bf16.gguf        16.4 GiB
hidream-o1-base-bf16.gguf       16.4 GiB
SHA256SUMS                      279 B
README.md                       model card (repo root)
LICENSE                         MIT, attributed to HiDream-ai
PROVENANCE.md                   per-artifact source + revision
```

To publish under a different namespace, override explicitly:

```bash
HF_REPO=other/o1.c-models ./release/hf_upload/upload.sh
```

Then update `hf_release_repo` and every `download_url` in
`models/manifest.json` to match, and re-verify the downloader.

## 4. Upstream provenance (verified)

All three source repositories exist and declare **MIT**:

| Source repository | HTTP | License | Revision pinned |
|-------------------|------|---------|-----------------|
| `HiDream-ai/HiDream-O1-Image-Dev-2604` | 200 | MIT | `b6acc2fe452b3120430620dc4354fa442ee081ea` |
| `HiDream-ai/HiDream-O1-Image-Dev`      | 200 | MIT | `c0bada0e15c54a9f96a6d1ecc35575b32bc21544` |
| `HiDream-ai/HiDream-O1-Image`          | 200 | MIT | `0b0901d99f200389e138c61946af1185f5f49a13` |

Upstream ships no standalone `LICENSE` file; the MIT text is reproduced in
`release/hf_upload/LICENSE` with attribution to HiDream-ai.

## 5. Upload payload (staged in `release/hf_upload/`)

| File | Purpose |
|------|---------|
| `README.md` | Model card: GGUF conversions for o1.c, credits **HiDream-ai**, links the engine repo <https://github.com/luca-saggese/o1.c>, states it does not replace upstream, provenance table, Q4 marked unavailable |
| `LICENSE` | MIT text, attributed to HiDream-ai |
| `PROVENANCE.md` | Per-artifact source repo + immutable revision, conversion semantics, reproduction steps |
| `upload.sh` | Credential-gated uploader (see below) |

Artifacts (built by R2, in `release/models/`, git-ignored):

| Artifact | Size | SHA256 |
|----------|------|--------|
| `hidream-o1-dev-2604-bf16.gguf` | 17 609 841 152 B | `7c405035a9225b721ba19b1a1f4e7f1cb937ddea7dbb7207e5ad677c3e0a7f5d` |
| `hidream-o1-dev-bf16.gguf`      | 17 609 841 152 B | `4b1141733ec5a99fcbc2994bf720e8ef6ade31ec3e9378e471e73f3af6ce8fae` |
| `hidream-o1-base-bf16.gguf`     | 17 609 841 152 B | `07f5efb347cfd9264c8dadd3848bb8c6053633b37a05b0680d920a6409de011b` |
| `SHA256SUMS` | 279 B | (checksums of the three above) |

Re-verified after staging:

```text
$ (cd release/models && sha256sum -c SHA256SUMS)
hidream-o1-dev-2604-bf16.gguf: OK
hidream-o1-dev-bf16.gguf: OK
hidream-o1-base-bf16.gguf: OK
```

## 6. `upload.sh` behaviour (verified)

The uploader is deliberately fail-closed:

1. **Target resolution** — reads `hf_release_repo` from `models/manifest.json`;
   falls back to `saggeseluca/o1.c-models`; `HF_REPO` overrides.
2. **Credential gate** — if neither `HF_TOKEN` nor `~/.cache/huggingface/token`
   exists, it prints an actionable message and exits 2 *before* any network
   call. Verified.
3. **Auth verification** — if a token is present but `hf auth whoami` fails,
   it aborts. Verified with a fake token.
4. **Checksum gate** — runs `sha256sum -c SHA256SUMS` against the artifacts
   and refuses to upload on any mismatch. Verified: all three `OK`.
5. **`--dry-run`** — lists the payload and uploads nothing. Verified.
6. **Upload** — `hf repo create --exist-ok` then `hf upload` per file.
   Verified end to end against a stub `hf`.

Test matrix executed:

| Scenario | Result |
|----------|--------|
| No credentials | exits 2, "Upload is PENDING", no upload |
| Token present, invalid | aborts at `whoami` |
| Credentials + `--dry-run` | verifies checksums, lists 7 files, uploads nothing |
| Credentials + real path (stub `hf`) | repo create + uploads, exit 0 |

## 7. Post-upload verification

Once published, confirm the public path is live and matches the manifest:

```bash
curl -sI https://huggingface.co/saggeseluca/o1.c-models/resolve/main/hidream-o1-dev-2604-bf16.gguf
./scripts/download_model.sh dev-2604 --dir /tmp/dl-check
```

The downloader must print `OK` after SHA256 verification. Only then is R5
considered validated against the live release.

## 8. Verdict

| Requirement | Status |
|-------------|--------|
| Model card crediting HiDream-ai, stating non-replacement | **PASS** (staged) |
| Source repo + immutable revision for every artifact | **PASS** |
| License/provenance files | **PASS** (MIT) |
| Publish only validated artifacts | **PASS** (checksum-gated) |
| Target namespace resolved | **PASS** — `saggeseluca/o1.c-models` |
| Manifest/downloader/README point at the real path | **PASS** |
| Actual upload | **PENDING — no credentials in this environment** |

**R4: READY, UPLOAD PENDING.** The only remaining step is `hf auth login`
followed by `./release/hf_upload/upload.sh`.
