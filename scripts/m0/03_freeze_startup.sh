#!/usr/bin/env bash
# M0.3 — Freeze the startup oracle (startup-only; NO forward, NO generation).
# Usage: 03_freeze_startup.sh <dev|base>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-}"
log() { printf '[freeze] %s\n' "$*"; }
die() { printf '[freeze][ERROR] %s\n' "$*" >&2; exit 1; }

[ -n "${PROFILE}" ] || die "usage: $0 <dev|base>"
[ -f "${ROOT}/config/${PROFILE}.json" ] || die "missing config/${PROFILE}.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The oracle requires transformers 4.57.1 (pinned in the venv); the system
# python may carry an incompatible transformers. Use the pinned venv.
PYBIN="${ROOT}/.venv/bin/python"
[ -x "${PYBIN}" ] || die "pinned venv not found at ${PYBIN}"

"${PYBIN}" - "${ROOT}" "${PROFILE}" <<'PY'
import json, os, sys
root, profile = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "python"))

cfg = json.load(open(os.path.join(root, "config", profile + ".json")))
local_path = os.path.join(root, cfg["local_path"])

from tools.freeze_startup import freeze_startup
freeze_startup(root, profile, local_path, cfg)
PY
