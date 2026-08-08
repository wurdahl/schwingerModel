#!/usr/bin/env bash
# Builds libdwf.so from dwfApplyD.cu using the Blackwell-capable nvcc that ships
# with the nvidia-cuda-nvcc pip wheel (the system nvcc 12.4 cannot target sm_120).
# Run from this directory, inside the 'science' conda env:  ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

# prefer a system nvcc new enough for Blackwell (sm_120 needs CUDA >= 12.8);
# otherwise fall back to the nvcc bundled with the jax cuda pip wheels
sysNvccOk=false
if command -v nvcc >/dev/null; then
    ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
    [ "${ver:-0}" -ge 128 ] && sysNvccOk=true
fi
if $sysNvccOk; then
    NVCC=$(command -v nvcc)
else
    NVCC=$(python -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13', 'bin', 'nvcc'))")
fi
FFI_INC=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")

echo "nvcc: $NVCC"
"$NVCC" -O3 -shared -std=c++17 --compiler-options=-fPIC \
    -arch=sm_120 \
    -I"$FFI_INC" \
    -o libdwf.so dwfApplyD.cu

echo "built $(pwd)/libdwf.so"
