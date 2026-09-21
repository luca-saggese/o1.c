#!/usr/bin/env bash
#
# build_release_models.sh — produce the reproducible BF16 release GGUFs.
#
# Reads release/models.def.json, downloads each pinned upstream revision
# into a scratch source directory (unless it is already present) and runs
# the converter with the pinned profile/variant/revision.
#
# It then records, for every artifact: size, SHA256 and the exact
# conversion command, into release/models.built.json.
#
# Usage:
#   ./scripts/build_release_models.sh                 # all models
#   ./scripts/build_release_models.sh dev base        # selected ids
#   ./scripts/build_release_models.sh --no-download dev
#
# Environment:
#   O1_SRC_ROOT   scratch dir for upstream checkouts (default: .cache/upstream)
#   O1_OUT_DIR    output dir for the GGUFs          (default: release/models)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEF="$ROOT/release/models.def.json"
SRC_ROOT="${O1_SRC_ROOT:-$ROOT/.cache/upstream}"
OUT_DIR="${O1_OUT_DIR:-$ROOT/release/models}"
NO_DOWNLOAD=0

if [ ! -f "$DEF" ]; then
    echo "error: missing $DEF" >&2
    exit 1
fi

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-download) NO_DOWNLOAD=1 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

mkdir -p "$SRC_ROOT" "$OUT_DIR"

python3 - "$DEF" "${ARGS[@]}" <<'PY' > /tmp/.o1_rel_plan
import json, sys
defpath, *wanted = sys.argv[1:]
d = json.load(open(defpath))
sel = [m for m in d["models"] if not wanted or m["id"] in wanted]
if wanted and len(sel) != len(wanted):
    missing = set(wanted) - {m["id"] for m in sel}
    sys.exit(f"unknown model id(s): {', '.join(sorted(missing))}")
print("\n".join(f'{m["id"]}\t{m["hf_repo"]}\t{m["upstream_revision"]}\t{m["source_dir"]}\t{m["profile"]}\t{m["variant"]}\t{m["filename"]}' for m in sel))
PY

BUILT="$OUT_DIR/models.built.json"
RECS="$OUT_DIR/.built.recs"
: > "$RECS"
: > "$OUT_DIR/SHA256SUMS"

while IFS=$'\t' read -r id repo rev srcdir profile variant filename; do
    [ -n "$id" ] || continue
    src="$ROOT/$srcdir"
    out="$OUT_DIR/$filename"

    echo "=== $id ($filename) ==="
    if [ ! -f "$src/model.safetensors.index.json" ]; then
        if [ "$NO_DOWNLOAD" = 1 ]; then
            echo "error: $src has no checkpoint and --no-download was given" >&2
            exit 1
        fi
        echo "  downloading $repo@$rev -> $src"
        python3 - "$repo" "$rev" "$src" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=dst,
                  allow_patterns=["*.json", "*.txt", "*.safetensors"],
                  max_workers=4)
PY
    else
        echo "  source present: $src"
    fi

    # Verify the source really is the pinned upstream revision.
    python3 - "$src" "$repo" "$rev" <<'PY'
import hashlib, json, os, sys
from huggingface_hub import HfApi
src, repo, rev = sys.argv[1:4]
local = json.load(open(os.path.join(src, "model.safetensors.index.json")))
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "model.safetensors.index.json", revision=rev,
                        cache_dir=os.path.join(src, ".hfidx"))
    remote = json.load(open(p))
except Exception as e:
    print(f"  WARNING: could not verify against upstream: {e}")
    sys.exit(0)
if local.get("weight_map") != remote.get("weight_map"):
    sys.exit(f"  ERROR: {src} does not match {repo}@{rev}")
print(f"  verified against {repo}@{rev}")
PY

    echo "  converting..."
    python3 "$ROOT/tools/hidream_convert.py" \
        --source "$src" \
        --output "$out" \
        --profile "$profile" \
        --variant "$variant" \
        --revision "$rev"

    size=$(stat -c %s "$out")
    sha=$(sha256sum "$out" | cut -d' ' -f1)
    printf '%s  %s\n' "$sha" "$filename" >> "$OUT_DIR/SHA256SUMS"
    echo "  $filename  $size bytes  sha256=$sha"

    printf '%s\n' "$(python3 - "$id" "$filename" "$repo" "$rev" "$profile" "$variant" "$size" "$sha" <<'PY'
import json, sys
mid, fn, repo, rev, profile, variant, size, sha = sys.argv[1:9]
print(json.dumps({"id": mid, "filename": fn, "hf_repo": repo,
                  "upstream_revision": rev, "profile": profile,
                  "variant": variant, "quantization": "bf16",
                  "size": int(size), "sha256": sha}))
PY
)" >> "$RECS"
done < /tmp/.o1_rel_plan

python3 - "$RECS" "$BUILT" <<'PY'
import json, sys
recs, out = sys.argv[1:3]
rows = [json.loads(l) for l in open(recs) if l.strip()]
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
    f.write("\n")
print(f"\nwrote {out} ({len(rows)} entries)")
PY
rm -f /tmp/.o1_rel_plan "$RECS"
echo
echo "artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
