# Environment setup (R6)

o1.c builds against the CUDA toolkit, cuBLAS/cuBLASLt and cuDNN. This page
explains how those are detected, how to override them, and how to install
cuDNN on a supported Linux system.

## 1. Quick path

```bash
source scripts/env.sh     # resolve and export CUDA_HOME / CUDNN_HOME / LD_LIBRARY_PATH
make
```

`scripts/env.sh` is safe to source repeatedly and is idempotent: it
de-duplicates `LD_LIBRARY_PATH` and does not prepend the same directory twice.
Set `O1_ENV_QUIET=1` to suppress its summary output.

To see a full diagnosis instead of just the resolved values:

```bash
./scripts/setup.sh
```

`setup.sh` checks the GPU, compute capability, driver, `nvcc`/CUDA headers,
`libcudart`, cuBLAS, cuBLASLt, cuDNN, the host compiler, `make` and the
vendored `cudnn-frontend` headers. It **only reports** — it never installs
packages or modifies the system. It exits `0` when everything required is
present and `1` otherwise, printing actionable errors for what is missing.

## 2. What is detected, and where

Both the Makefile and the scripts probe the same standard locations, so a
plain `make` works without any exports on a normal install.

| Component | Search order |
|-----------|--------------|
| CUDA | `$CUDA_HOME`, then `/usr/local/cuda`, `/usr/local/cuda-*`, `/opt/cuda`, `/usr`, then the directory containing `nvcc` |
| cuDNN | `$CUDNN_HOME`, `$CUDA_PATH`, `/usr/local/cudnn`, `/usr/local/cudnn-*`, `/usr/local/cuda`, `/opt/cudnn`, `/usr`, `/usr/local`, then the `nvidia-cudnn` wheel layout under `python3*/{site,dist}-packages` |

**No absolute machine-specific path is hard-coded anywhere.** The previous
`Makefile` pinned `CUDNN_HOME` to one developer's Python `site-packages`
directory; that has been replaced by detection plus an explicit override.

## 3. Overrides

Any of these work and take precedence over detection:

```bash
# one-off
make CUDNN_HOME=/opt/cudnn

# persistent, for the shell
export CUDNN_HOME=/opt/cudnn
export CUDA_HOME=/usr/local/cuda-13.0
source scripts/env.sh
```

`CUDA_HOME` and `CUDNN_HOME` are the only overrides needed. `env.sh` then
derives `PATH` and `LD_LIBRARY_PATH` from them.

## 4. Installing cuDNN on Linux

cuDNN 9.x is required. Two supported methods:

### a) NVIDIA Python wheel (no root, simplest)

```bash
pip install nvidia-cudnn-cu13     # for CUDA 13
# or
pip install nvidia-cudnn-cu12     # for CUDA 12
```

The wheel installs headers and `libcudnn.so.9` under
`<python>/site-packages/nvidia/cudnn/`. Auto-detection finds this layout, so
nothing else is required. If you have several Python installations and want to
pin one:

```bash
export CUDNN_HOME=~/.local/lib/python3.12/site-packages/nvidia/cudnn
```

Note that only the shared library and headers are used at build/run time; the
o1.c runtime does not import Python.

### b) NVIDIA native tarball (system-wide)

Download the cuDNN 9.x Linux tarball for your CUDA major version from NVIDIA,
then:

```bash
tar -xzf cudnn-linux-x86_64-9.*_cuda13-archive.tar.xz -C /usr/local
mv /usr/local/cudnn-linux-x86_64-9.*_cuda13-archive /usr/local/cudnn
export CUDNN_HOME=/usr/local/cudnn
```

`/usr/local/cudnn` is in the default search path, so detection then works
without the export.

### Verifying

```bash
./scripts/setup.sh | grep -A3 cuDNN
```

should report the cuDNN version, the header directory and the directory
containing `libcudnn.so*`.

## 5. Runtime library path

The CLI and server link `libcudnn.so.9` dynamically. If it lives outside the
system loader path (the wheel layout does), export the search path:

```bash
source scripts/env.sh
```

or, equivalently and explicitly:

```bash
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$LD_LIBRARY_PATH"
```

Confirm the resolved libraries with:

```bash
ldd build/hidream | grep -E 'cudnn|cublas'
```

## 6. Requirements summary

| Requirement | Minimum |
|-------------|---------|
| GPU | NVIDIA, compute capability 12.1 for the validated build (sm_121) |
| Driver | capable of running the installed CUDA toolkit |
| CUDA toolkit | 13.0 validated (`nvcc`, headers, `libcudart`) |
| cuBLAS / cuBLASLt | shipped with the CUDA toolkit |
| cuDNN | 9.x (9.20 validated) |
| Compiler | C11 + C++17 host compiler (gcc 13.3 validated) |
| make | GNU make |

## 7. Gate result

```
R6 environment/setup: PASS
```
