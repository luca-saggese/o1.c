#!/usr/bin/env bash
# o1.c setup and environment validation.
#
# Detects and validates the toolchain and GPU libraries required to build and
# run o1.c, then prints the exact exports needed. It never installs anything
# and never modifies system state: it only reports.
#
# Usage:
#   ./scripts/setup.sh            # detect and validate
#   CUDNN_HOME=/path ./scripts/setup.sh
#
# Exit code 0 if everything required was found, 1 otherwise.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FAIL=0
WARN=0
ok()   { printf '  \033[32m[ OK ]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=1; }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$*"; WARN=1; }

echo "o1.c setup check"
echo "================"
echo

# --------------------------------------------------------------- GPU
echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        nvidia-smi -L | sed 's/^/       /'
        CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        ok "driver $DRV, compute capability ${CC:-unknown}"
    else
        bad "nvidia-smi present but no GPU detected"
    fi
else
    bad "nvidia-smi not found (NVIDIA driver not installed?)"
fi
echo

# --------------------------------------------------------------- compiler
echo "Host compiler"
if command -v cc >/dev/null 2>&1; then
    ok "$(cc --version | head -1)"
else
    bad "no C compiler (cc) found; install gcc"
fi
if command -v make >/dev/null 2>&1; then
    ok "$(make --version | head -1)"
else
    bad "make not found"
fi
echo

# --------------------------------------------------------------- CUDA
echo "CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    ok "nvcc $NVCC_VER ($(command -v nvcc))"
else
    bad "nvcc not found; install the CUDA toolkit"
    NVCC_VER=""
fi

if [ -z "${CUDA_HOME:-}" ]; then
    for d in /usr/local/cuda /usr/local/cuda-* /opt/cuda /usr; do
        if [ -f "$d/include/cuda_runtime.h" ]; then CUDA_HOME="$d"; break; fi
    done
fi
if [ -n "${CUDA_HOME:-}" ] && [ -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
    ok "CUDA headers: $CUDA_HOME/include"
else
    bad "CUDA headers not found (cuda_runtime.h); set CUDA_HOME"
fi
if [ -n "${CUDA_HOME:-}" ] && { [ -e "$CUDA_HOME/lib64/libcudart.so" ] || \
        ls "$CUDA_HOME"/lib64/libcudart.so.* >/dev/null 2>&1; }; then
    ok "libcudart found in $CUDA_HOME/lib64"
else
    warn "libcudart.so not found under \$CUDA_HOME/lib64"
fi
echo

# --------------------------------------------------------------- cuBLAS
echo "cuBLAS / cuBLASLt"
_find_lib() {  # name -> echoes first directory containing it
    local n="$1"
    for d in "${CUDA_HOME:-}/lib64" /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu \
             /usr/local/lib /usr/lib; do
        [ -d "$d" ] || continue
        if ls "$d/$n".so* >/dev/null 2>&1; then echo "$d"; return 0; fi
    done
    return 1
}
if d=$(_find_lib libcublas); then ok "libcublas found in $d"; else bad "libcublas not found"; fi
if d=$(_find_lib libcublasLt); then ok "libcublasLt found in $d"; else bad "libcublasLt not found"; fi
echo

# --------------------------------------------------------------- cuDNN
echo "cuDNN"
if [ -z "${CUDNN_HOME:-}" ]; then
    for d in "${CUDA_PATH:-}" /usr/local/cudnn /usr/local/cudnn-* \
             /usr/local/cuda /opt/cudnn /usr /usr/local; do
        [ -n "$d" ] || continue
        if [ -f "$d/include/cudnn.h" ] || [ -f "$d/include/cudnn_version.h" ]; then
            CUDNN_HOME="$d"; break
        fi
    done
fi
if [ -z "${CUDNN_HOME:-}" ]; then
    for d in $(ls -d /usr/local/lib/python3*/dist-packages/nvidia/cudnn \
                      /usr/local/lib/python3*/site-packages/nvidia/cudnn \
                      "$HOME"/.local/lib/python3*/site-packages/nvidia/cudnn \
                      /usr/lib/python3*/dist-packages/nvidia/cudnn 2>/dev/null); do
        if [ -f "$d/include/cudnn.h" ] || [ -f "$d/include/cudnn_version.h" ]; then
            CUDNN_HOME="$d"; break
        fi
    done
fi

CUDNN_LIBDIR=""
if [ -n "${CUDNN_HOME:-}" ]; then
    if [ -f "$CUDNN_HOME/include/cudnn_version.h" ]; then
        V=$(grep -E '^#define CUDNN_(MAJOR|MINOR|PATCHLEVEL)' "$CUDNN_HOME/include/cudnn_version.h" \
            | awk '{print $3}' | paste -sd. -)
        ok "cuDNN $V headers: $CUDNN_HOME/include"
    elif [ -f "$CUDNN_HOME/include/cudnn.h" ]; then
        ok "cuDNN headers: $CUDNN_HOME/include"
    fi
    for d in "$CUDNN_HOME/lib" "$CUDNN_HOME/lib64"; do
        if ls "$d"/libcudnn.so* >/dev/null 2>&1; then CUDNN_LIBDIR="$d"; break; fi
    done
fi

if [ -z "$CUDNN_LIBDIR" ]; then
    bad "cuDNN library not found; install cuDNN or set CUDNN_HOME"
    cat <<'EOF'
       A supported installation method (no root required):

         pip install nvidia-cudnn-cu13
         # or, for CUDA 12: pip install nvidia-cudnn-cu12

       Then either rely on auto-detection, or point at it explicitly:

         CUDNN_HOME=~/.local/lib/python3.12/site-packages/nvidia/cudnn

       Native installs also work: unzip the cuDNN tar into /usr/local/cudnn
       and set CUDNN_HOME=/usr/local/cudnn.
EOF
else
    ok "libcudnn found in $CUDNN_LIBDIR"
fi

if [ ! -d "$ROOT/third_party/cudnn-frontend/include" ]; then
    bad "third_party/cudnn-frontend missing (required headers)"
else
    ok "cudnn-frontend headers present"
fi
echo

# --------------------------------------------------------------- summary
echo "Resolved environment"
echo "--------------------"
[ -n "${CUDA_HOME:-}" ]  && echo "  export CUDA_HOME=$CUDA_HOME"
[ -n "${CUDNN_HOME:-}" ] && echo "  export CUDNN_HOME=$CUDNN_HOME"
echo
echo "  Or simply:  source scripts/env.sh"
echo

if [ "$FAIL" = "1" ]; then
    echo "Result: FAIL — required components missing (see above)."
    exit 1
fi
if [ "$WARN" = "1" ]; then
    echo "Result: PASS with warnings."
else
    echo "Result: PASS — environment looks good. Next: make"
fi
