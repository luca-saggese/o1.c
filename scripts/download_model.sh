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

# ---- manifest parsing (no Python required) ----------------------------------
# models/manifest.json is pretty-printed with 4-space object indentation, so a
# small awk reader is enough. Keeping this dependency-free is deliberate: the
# download path must work on a machine without Python or PyTorch.
# Only the top-level "models" array is considered; the "unavailable" list is
# reported separately.
mf_obj() { # $1 = model id -> prints the manifest object for that id
    awk -v mid="$1" '
        /^  "models": \[/ { insec = 1; next }
        insec && /^  \]/  { insec = 0 }
        !insec            { next }
        /^    \{/ { inobj = 1; buf = "" }
        inobj     { buf = buf $0 "\n" }
        /^    \}/ {
            if (inobj && buf ~ ("\"id\": \"" mid "\"")) { printf "%s", buf; exit }
            inobj = 0
        }
    ' "$MANIFEST"
}

mf_str() { mf_obj "$1" | sed -n "s/^[[:space:]]*\"$2\":[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1; }
mf_num() { mf_obj "$1" | sed -n "s/^[[:space:]]*\"$2\":[[:space:]]*\([0-9][0-9]*\).*/\1/p" | head -1; }
mf_bool() { mf_obj "$1" | sed -n "s/^[[:space:]]*\"$2\":[[:space:]]*\(true\|false\).*/\1/p" | head -1; }

mf_all_ids() { # ids in manifest order, models array only
    awk '/^  "models": \[/ { insec = 1; next }
         insec && /^  \]/  { insec = 0 }
         insec && match($0, /"id": "[^"]*"/) { print substr($0, RSTART + 7, RLENGTH - 8) }' "$MANIFEST"
}

# ---- list -------------------------------------------------------------------
if [ -z "$MODEL_ID" ] || [ "$MODEL_ID" = "list" ]; then
    printf 'Available models (quantisation: bf16):\n\n'
    printf '  %-12s %-32s %8s  %s\n' "ID" "NAME" "SIZE" "NOTES"
    for id in $(mf_all_ids); do
        name="$(mf_str "$id" display_name)"
        size="$(mf_num "$id" size)"
        note=""
        [ "$(mf_bool "$id" recommended)" = "true" ] && note="recommended"
        if [ -n "$size" ] && [ "$size" -gt 0 ] 2>/dev/null; then
            gb="$(awk -v s="$size" 'BEGIN { printf "%.1fG", s/1e9 }')"
        else
            gb="-"
        fi
        printf '  %-12s %-32s %8s  %s\n' "$id" "$name" "$gb" "$note"
    done
    printf '\nNot yet available:\n'
    printf '  dev-q4        (q4) - unavailable\n'
    printf '  dev-2604-q4   (q4) - unavailable\n'
    printf '  base-q4       (q4) - unavailable\n'
    printf '\nUsage: ./scripts/download_model.sh <id> [bf16]\n'
    exit 0
fi

# ---- resolve the entry ------------------------------------------------------
if [ "$QUANT" != "bf16" ]; then
    if grep -q "\"quantization\": \"$QUANT\"" "$MANIFEST"; then
        die "'$QUANT' models are not available yet.
       The o1.c runtime currently supports only F32/F16/BF16 GGUF tensors."
    fi
    die "unknown quantisation '$QUANT' (only bf16 is available)"
fi

if ! mf_all_ids | grep -qx "$MODEL_ID"; then
    die "unknown model id '$MODEL_ID'.
       available: $(mf_all_ids | tr '\n' ' ')"
fi

URL="$(mf_str "$MODEL_ID" download_url)"
SHA="$(mf_str "$MODEL_ID" sha256)"
SIZE="$(mf_num "$MODEL_ID" size)"
FILENAME="$(mf_str "$MODEL_ID" filename)"

[ -n "$URL" ] && [ -n "$SHA" ] && [ -n "$FILENAME" ] \
    || die "could not resolve model '$MODEL_ID' from $MANIFEST"

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
