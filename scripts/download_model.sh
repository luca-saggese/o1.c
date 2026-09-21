#!/usr/bin/env bash
#
# download_model.sh — fetch a validated o1.c model from the release manifest.
#
# Usage:
#   ./scripts/download_model.sh dev-2604          # recommended Dev-2604 BF16
#   ./scripts/download_model.sh dev               # plain Dev BF16
#   ./scripts/download_model.sh base              # Base/Full BF16
#   ./scripts/download_model.sh list              # show what is available
#   ./scripts/download_model.sh dev-2604 q4       # quantisation selector
#
# Options:
#   --dir DIR     destination directory (default: models)
#   --force       re-download even if the file already exists and verifies
#
# The script reads models/manifest.json, downloads with resume support and a
# progress bar, then verifies SHA256 and fails loudly on any mismatch. It needs
# only curl (or wget) and sha256sum — no Python or PyTorch stack.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/models/manifest.json"
DEST="$ROOT/models"
FORCE=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# ---- argument parsing -------------------------------------------------------
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)   DEST="$2"; shift 2 ;;
        --dir=*) DEST="${1#*=}"; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option: $1" ;;
        *) POS+=("$1"); shift ;;
    esac
done

[ -f "$MANIFEST" ] || die "missing manifest: $MANIFEST"
command -v sha256sum >/dev/null || die "sha256sum is required but not installed"

MODEL_ID="${POS[0]:-}"
QUANT="${POS[1]:-bf16}"

# ---- list -------------------------------------------------------------------
if [ -z "$MODEL_ID" ] || [ "$MODEL_ID" = "list" ]; then
    python3 - "$MANIFEST" <<'PY' 2>/dev/null || awk '/"id"/{print}' "$MANIFEST"
import json, sys
d = json.load(open(sys.argv[1]))
print("Available models (quantisation: bf16):\n")
print(f"  {'ID':<12} {'NAME':<32} {'SIZE':>8}  NOTES")
for m in d["models"]:
    gb = m["size"] / 1e9
    note = "recommended" if m.get("recommended") else ""
    print(f"  {m['id']:<12} {m['display_name']:<32} {gb:>6.1f}G  {note}")
print("\nNot yet available:")
for u in d.get("unavailable", []):
    print(f"  {u['id']:<12} ({u['quantization']}) — {u.get('status','unavailable')}")
print("\nUsage: ./scripts/download_model.sh <id> [bf16]")
PY
    exit 0
fi

# ---- resolve the entry ------------------------------------------------------
read -r URL SHA SIZE FILENAME < <(python3 - "$MANIFEST" "$MODEL_ID" "$QUANT" <<'PY'
import json, sys
path, mid, quant = sys.argv[1:4]
d = json.load(open(path))
if quant != "bf16":
    for u in d.get("unavailable", []):
        if u["quantization"] == quant:
            sys.exit(f"error: '{quant}' models are not available yet.\n"
                     f"       {u.get('reason','')}")
    sys.exit(f"error: unknown quantisation '{quant}' (only bf16 is available)")
for m in d["models"]:
    if m["id"] == mid:
        print(m["download_url"], m["sha256"], m["size"], m["filename"])
        break
else:
    ids = ", ".join(m["id"] for m in d["models"])
    sys.exit(f"error: unknown model id '{mid}'.\n       available: {ids}")
PY
) || exit 1

[ -n "${URL:-}" ] || die "could not resolve model '$MODEL_ID'"

mkdir -p "$DEST"
OUT="$DEST/$FILENAME"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

# ---- already present? -------------------------------------------------------
if [ -f "$OUT" ] && [ "$FORCE" = 0 ]; then
    printf 'checking existing %s ... ' "$OUT"
    if echo "$SHA  $OUT" | sha256sum -c --status 2>/dev/null; then
        echo "OK (already downloaded)"
        printf '\nModel ready. Run:\n\n  ./build/hidream --model-path %s --prompt "a teapot" --output out.png\n\n' "$OUT"
        exit 0
    fi
    echo "corrupt or incomplete; re-downloading"
fi

# ---- download ---------------------------------------------------------------
echo "Downloading $FILENAME ($(human "$SIZE"))"
echo "  from $URL"
echo "  to   $OUT"

dl() {
    if command -v curl >/dev/null; then
        if [ -s "$OUT.part" ]; then
            if curl -L --fail --retry 5 --retry-delay 3 --continue-at - \
                    --progress-bar -o "$OUT.part" "$URL"; then
                return 0
            fi
            echo "  (server cannot resume; restarting download)" >&2
            rm -f "$OUT.part"
        fi
        curl -L --fail --retry 5 --retry-delay 3 \
             --progress-bar -o "$OUT.part" "$URL"
    elif command -v wget >/dev/null; then
        wget -c --progress=bar:force -O "$OUT.part" "$URL"
    else
        die "neither curl nor wget is available"
    fi
}

mkdir -p "$(dirname "$OUT")"
if ! dl; then
    die "download failed (partial file left at $OUT.part for resume)"
fi
mv "$OUT.part" "$OUT"

# ---- verify -----------------------------------------------------------------
echo
printf 'verifying SHA256 ... '
ACTUAL="$(sha256sum "$OUT" | cut -d' ' -f1)"
if [ "$ACTUAL" != "$SHA" ]; then
    rm -f "$OUT"
    die "checksum mismatch for $FILENAME
       expected $SHA
       actual   $ACTUAL
       the file was removed; please retry"
fi
echo "OK"

SIZE_ON_DISK="$(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT")"
if [ "$SIZE_ON_DISK" != "$SIZE" ]; then
    echo "warning: size is $SIZE_ON_DISK, manifest says $SIZE" >&2
fi

printf '\nModel ready. Run:\n\n  ./build/hidream --model-path %s --prompt "a teapot" --output out.png\n\n' "$OUT"
