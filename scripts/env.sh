#!/usr/bin/env bash
# o1.c environment resolution.
#
# Source this file to export the environment needed to build and run o1.c:
#
#     source scripts/env.sh
#
# It resolves CUDA_HOME and CUDNN_HOME from standard locations (or honours
# existing overrides) and exports PATH and LD_LIBRARY_PATH accordingly.
#
# It is safe to source repeatedly.

# ---------------------------------------------------------------- CUDA_HOME
if [ -z "${CUDA_HOME:-}" ]; then
    for d in /usr/local/cuda /usr/local/cuda-* /opt/cuda /usr; do
        if [ -f "$d/include/cuda_runtime.h" ]; then
            CUDA_HOME="$d"
            break
        fi
    done
fi
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export CUDA_HOME

# --------------------------------------------------------------- CUDNN_HOME
if [ -z "${CUDNN_HOME:-}" ]; then
    for d in "${CUDA_PATH:-}" /usr/local/cudnn /usr/local/cudnn-* \
             /usr/local/cuda /opt/cudnn /usr /usr/local; do
        [ -n "$d" ] || continue
        if [ -f "$d/include/cudnn.h" ] || [ -f "$d/include/cudnn_version.h" ]; then
            CUDNN_HOME="$d"
            break
        fi
    done
fi
if [ -z "${CUDNN_HOME:-}" ]; then
    for d in $(ls -d /usr/local/lib/python3*/dist-packages/nvidia/cudnn \
                      /usr/local/lib/python3*/site-packages/nvidia/cudnn \
                      "$HOME"/.local/lib/python3*/site-packages/nvidia/cudnn \
                      /usr/lib/python3*/dist-packages/nvidia/cudnn 2>/dev/null); do
        if [ -f "$d/include/cudnn.h" ] || [ -f "$d/include/cudnn_version.h" ]; then
            CUDNN_HOME="$d"
            break
        fi
    done
fi
export CUDNN_HOME

# ----------------------------------------------------------------- PATH etc.
if [ -n "${CUDA_HOME:-}" ]; then
    case ":$PATH:" in
        *":$CUDA_HOME/bin:"*) ;;
        *) PATH="$CUDA_HOME/bin:$PATH" ;;
    esac
fi
export PATH

_ld=""
if [ -n "${CUDA_HOME:-}" ]; then
    [ -d "$CUDA_HOME/lib64" ] && _ld="$CUDA_HOME/lib64"
    [ -d "$CUDA_HOME/lib" ] && _ld="${_ld:+$_ld:}$CUDA_HOME/lib"
fi
if [ -n "${CUDNN_HOME:-}" ]; then
    [ -d "$CUDNN_HOME/lib" ] && _ld="${_ld:+$_ld:}$CUDNN_HOME/lib"
    [ -d "$CUDNN_HOME/lib64" ] && _ld="${_ld:+$_ld:}$CUDNN_HOME/lib64"
fi
if [ -n "$_ld" ]; then
    LD_LIBRARY_PATH="${_ld}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Normalise: drop empty entries and duplicates, preserving order.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    _norm=""
    for _d in $(printf '%s' "$LD_LIBRARY_PATH" | tr ':' ' '); do
        [ -n "$_d" ] || continue
        case ":$_norm:" in
            *":$_d:"*) ;;
            *) _norm="${_norm:+$_norm:}$_d" ;;
        esac
    done
    LD_LIBRARY_PATH="$_norm"
    unset _norm
fi
export LD_LIBRARY_PATH
unset _ld _d

if [ "${O1_ENV_QUIET:-0}" != "1" ]; then
    echo "o1.c environment:"
    echo "  CUDA_HOME   = ${CUDA_HOME:-<not found>}"
    echo "  CUDNN_HOME  = ${CUDNN_HOME:-<not found>}"
    echo "  LD_LIBRARY_PATH = ${LD_LIBRARY_PATH:-<empty>}"
fi
