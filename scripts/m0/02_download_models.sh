#!/usr/bin/env bash
# M0.2 — Download pinned model weights into ignored project storage.
# Usage: 02_download_models.sh <dev|base>
# Downloads into models/<profile> using an immutable HF revision from
# config/models.lock (or config/<profile>.json immutable_revision).
# Fails closed: refuses to download from a moving ref (main/dev/latest).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-}"
log() { printf '[models] %s\n' "$*"; }
die() { printf '[models][ERROR] %s\n' "$*" >&2; exit 1; }

[ -n "${PROFILE}" ] || die "usage: $0 <dev|base>"
[ -f "${ROOT}/config/${PROFILE}.json" ] || die "missing config/${PROFILE}.json"

REPO="$(python3 -c "import json,sys;print(json.load(open('${ROOT}/config/${PROFILE}.json'))['hf_repo'])")"
REV="$(python3 -c "import json,sys;print(json.load(open('${ROOT}/config/${PROFILE}.json'))['immutable_revision'])")"
DEST="${ROOT}/models/${PROFILE}"

# Fail closed on moving refs.
case "${REV}" in
  ""|main|dev|latest|*"latest"*)
    die "refusing to download from moving ref: profile=${PROFILE} revision='${REV}'" ;;
esac
case "${REV}" in
  *[!0-9a-f]*) die "revision is not a hex commit for profile=${PROFILE}: '${REV}'" ;;
esac

log "Downloading ${REPO} @ ${REV} -> ${DEST}"
python3 - <<PY
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO}",
    revision="${REV}",
    local_dir="${DEST}",
    local_dir_use_symlinks=False,
)
PY

log "Download complete. Verifying local_files_only=True resolution:"
python3 - <<PY
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("${DEST}", local_files_only=True)
print("model_type:", getattr(cfg, "model_type", None))
print("architectures:", getattr(cfg, "architectures", None))
PY
