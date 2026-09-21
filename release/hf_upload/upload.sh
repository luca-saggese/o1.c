#!/usr/bin/env bash
# Upload the o1.c release artifacts to the Hugging Face release repository.
#
# This script never invents success: it verifies credentials first, and it
# refuses to upload anything that does not match the recorded SHA256.
#
# Usage:
#   ./release/hf_upload/upload.sh [--dry-run]
#   HF_REPO=other/o1.c-models ./release/hf_upload/upload.sh
#
# Requires: huggingface_hub (hf CLI) and valid credentials (hf auth login,
# or HF_TOKEN in the environment).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPLOAD_DIR="$ROOT/release/hf_upload"
ART_DIR="$ROOT/release/models"
DEF="$ROOT/release/models.def.json"
MANIFEST="$ROOT/models/manifest.json"

# Default target comes from models/manifest.json (single source of truth),
# falling back to the published namespace.
if [ -z "${HF_REPO:-}" ] && [ -f "$MANIFEST" ]; then
    HF_REPO="$(sed -n 's/.*"hf_release_repo"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$MANIFEST" | head -1)"
fi
HF_REPO="${HF_REPO:-saggeseluca/o1.c-models}"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- credentials
echo "==> Checking Hugging Face credentials"
if [ -n "${HF_TOKEN:-}" ]; then
    echo "    HF_TOKEN present in environment"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "    token file present: ~/.cache/huggingface/token"
else
    cat >&2 <<'EOF'
error: no Hugging Face credentials found.

Upload is PENDING. To publish, authenticate first:

    hf auth login
    # or: export HF_TOKEN=hf_xxx

Then re-run this script. Nothing has been uploaded.
EOF
    exit 2
fi

command -v hf >/dev/null || die "'hf' CLI not found (pip install huggingface_hub)"

if hf auth whoami >/dev/null 2>&1; then
    echo "    authenticated as: $(hf auth whoami 2>/dev/null | head -1)"
else
    die "credentials present but 'hf auth whoami' failed; not uploading"
fi

[ -n "$HF_REPO" ] || die "set HF_REPO=owner/repo (e.g. HF_REPO=saggeseluca/o1.c-models)"
echo "    target repository: $HF_REPO"

# ------------------------------------------------------------- artifact check
echo "==> Verifying artifacts against SHA256SUMS"
[ -f "$ART_DIR/SHA256SUMS" ] || die "missing $ART_DIR/SHA256SUMS; run scripts/build_release_models.sh"
( cd "$ART_DIR" && sha256sum -c SHA256SUMS ) || die "artifact checksum mismatch; refusing to upload"

# --------------------------------------------------------------------- upload
FILES=()
for f in "$ART_DIR"/hidream-o1-*-bf16.gguf; do
    [ -e "$f" ] || die "no GGUF artifacts found in $ART_DIR"
    FILES+=("$f")
done
FILES+=("$ART_DIR/SHA256SUMS")
FILES+=("$UPLOAD_DIR/README.md")
FILES+=("$UPLOAD_DIR/LICENSE")
FILES+=("$UPLOAD_DIR/PROVENANCE.md")

echo "==> Files to upload to $HF_REPO"
for f in "${FILES[@]}"; do
    printf '    %-40s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done

if [ "$DRY_RUN" = "1" ]; then
    echo "==> --dry-run: nothing uploaded"
    exit 0
fi

echo "==> Creating repository if needed"
hf repo create "$HF_REPO" --repo-type model --exist-ok

echo "==> Uploading"
for f in "${FILES[@]}"; do
    base="$(basename "$f")"
    # The model card must be named README.md at the repo root.
    echo "    -> $base"
    hf upload "$HF_REPO" "$f" "$base" --repo-type model
done

echo
echo "==> Upload complete: https://huggingface.co/$HF_REPO"
echo "    Verify the published SHA256 values match models/manifest.json."
