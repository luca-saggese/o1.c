# R4 — Hugging Face release repository

**Gate status: PREPARED — upload PENDING (no credentials in this environment).**

Per the release-hardening rules, this gate does **not** invent success. All
release materials are staged and verified locally; publication requires
Hugging Face credentials that are absent here.

## 1. Credential check

| Check | Result |
|-------|--------|
| `HF_TOKEN` in environment | absent |
| `~/.cache/huggingface/token` | absent |
| `hf auth whoami` | `Error: Not logged in` |
| `hf` CLI available | yes (`~/.local/bin/hf`) |

Conclusion: **cannot authenticate**, therefore **cannot upload**. The upload
script detects this and exits with code 2 before touching the network.

## 2. Target repository — UNRESOLVED

The manifest placeholder `hf_release_repo = luca-saggese/o1.c-models`
**does not resolve**:

```text
GET https://huggingface.co/api/users/luca-saggese            -> 404
GET https://huggingface.co/api/models/luca-saggese/o1.c-models -> 401
```

The GitHub remote is `luca-saggese/o1.c`, but **no Hugging Face user or
organization named `luca-saggese` exists**. The real namespace must be
supplied by the maintainer before publication. The upload script takes it
from `HF_REPO`, so no code change is needed — only a decision.

**Action required:** confirm the HF owner (user or org) that will host the
release repository, then set `hf_release_repo` in `models/manifest.json` and
the `download_url` values accordingly.

## 3. Upstream provenance (verified)

All three source repositories exist and declare **MIT**:

| Source repository | HTTP | License | Revision pinned |
|-------------------|------|---------|-----------------|
| `HiDream-ai/HiDream-O1-Image-Dev-2604` | 200 | MIT | `b6acc2fe452b3120430620dc4354fa442ee081ea` |
| `HiDream-ai/HiDream-O1-Image-Dev`      | 200 | MIT | `c0bada0e15c54a9f96a6d1ecc35575b32bc21544` |
| `HiDream-ai/HiDream-O1-Image`          | 200 | MIT | `0b0901d99f200389e138c61946af1185f5f49a13` |

Upstream ships no standalone `LICENSE` file; the MIT text is reproduced in
`release/hf_upload/LICENSE` with attribution to HiDream-ai.

## 4. Upload payload (staged in `release/hf_upload/`)

| File | Purpose |
|------|---------|
| `README.md` | Model card: GGUF conversions for o1.c, credits **HiDream-ai**, states it does not replace upstream, provenance table, Q4 marked unavailable |
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

## 5. `upload.sh` behaviour (verified)

The uploader is deliberately fail-closed:

1. **Credential gate** — if neither `HF_TOKEN` nor `~/.cache/huggingface/token`
   exists, it prints an actionable message and exits 2 *before* any network
   call. Verified.
2. **Auth verification** — if a token is present but `hf auth whoami` fails,
   it aborts. Verified with a fake token.
3. **Checksum gate** — runs `sha256sum -c SHA256SUMS` against the artifacts
   and refuses to upload on any mismatch. Verified: all three `OK`.
4. **Explicit target** — requires `HF_REPO=owner/repo`; no hard-coded owner.
5. **`--dry-run`** — lists the payload and uploads nothing. Verified.
6. **Upload** — `hf repo create --exist-ok` then `hf upload` per file.
   Verified end to end against a stub `hf`.

Test matrix executed:

| Scenario | Result |
|----------|--------|
| No credentials | exits 2, "Upload is PENDING", no upload |
| Token present, invalid | aborts at `whoami` |
| Credentials + `--dry-run` | lists 7 files, uploads nothing |
| Credentials + real path (stub `hf`) | repo create + 7 uploads, exit 0 |

## 6. Publication procedure (for the maintainer)

```bash
hf auth login                       # or: export HF_TOKEN=hf_xxx
HF_REPO=<owner>/o1.c-models ./release/hf_upload/upload.sh --dry-run
HF_REPO=<owner>/o1.c-models ./release/hf_upload/upload.sh
```

Then update `models/manifest.json` (`hf_release_repo` and each `download_url`)
to the real namespace and re-verify `scripts/download_model.sh`.

## 7. Verdict

| Requirement | Status |
|-------------|--------|
| Model card crediting HiDream-ai, stating non-replacement | **PASS** (staged) |
| Source repo + immutable revision for every artifact | **PASS** |
| License/provenance files | **PASS** (MIT) |
| Publish only validated artifacts | **PASS** (checksum-gated) |
| Actual upload | **PENDING — no credentials** |
| Target namespace resolved | **BLOCKED — `luca-saggese` does not exist on HF** |

**R4: PREPARED, UPLOAD PENDING.** Two blockers for publication: absent
credentials, and an unconfirmed HF namespace.
