#!/usr/bin/env bash
# M0.1 — Checkout the official HiDream Python oracle into /python.
# The oracle is pinned by immutable commit SHA and treated as read-only.
# This script fails closed: it never falls back to a moving ref.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM_URL="https://github.com/HiDream-ai/HiDream-O1-Image.git"
UPSTREAM_BRANCH="dev"

# Optional explicit SHA override; otherwise resolve the branch SHA immutably.
PIN_SHA="${PIN_SHA:-}"
LOCK_FILE="${ROOT}/config/oracle.lock"

log() { printf '[oracle] %s\n' "$*"; }
die() { printf '[oracle][ERROR] %s\n' "$*" >&2; exit 1; }

[ -d "${ROOT}/python/.git" ] || {
  log "No /python checkout present; cloning upstream."
  git clone --no-checkout "${UPSTREAM_URL}" "${ROOT}/python"
}

if [ -z "${PIN_SHA}" ]; then
  PIN_SHA="$(git -C "${ROOT}/python" ls-remote "${UPSTREAM_URL}" "refs/heads/${UPSTREAM_BRANCH}" | cut -f1)"
fi

# Fail closed: a missing or non-hex SHA is fatal (never default to a branch name).
[ -n "${PIN_SHA}" ] || die "cannot resolve upstream revision"
case "${PIN_SHA}" in
  *[!0-9a-f]*) die "resolved SHA is not hex: '${PIN_SHA}'" ;;
esac
[ "${#PIN_SHA}" -eq 40 ] || die "resolved SHA is not 40 hex chars: '${PIN_SHA}'"

log "Pinning oracle to ${PIN_SHA} (branch ${UPSTREAM_BRANCH})."
git -C "${ROOT}/python" fetch --depth 1 origin "${PIN_SHA}"
git -C "${ROOT}/python" checkout --detach "${PIN_SHA}"

# Never patch the oracle; only record the immutable pin.
mkdir -p "$(dirname "${LOCK_FILE}")"
FREEZE_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "${LOCK_FILE}" <<EOF
{
  "upstream_url": "${UPSTREAM_URL}",
  "upstream_branch": "${UPSTREAM_BRANCH}",
  "upstream_commit_sha": "${PIN_SHA}",
  "freeze_date": "${FREEZE_DATE}"
}
EOF

log "Oracle checked out. SHA: ${PIN_SHA}"
git -C "${ROOT}/python" status --short
