"""Loader for the fused CUDA DWF operator kernel (cuda/dwfApplyD.cu).

The kernel is an XLA custom call registered through jax.ffi, so it composes with
jit / vmap / lax.scan like any jax op. Build it with cuda/build.sh; if libdwf.so
is absent, hmc_gpu falls back to the pure-XLA roll implementation.
"""
import ctypes
import os

import numpy as np
import jax
import jax.numpy as jnp

_SO_PATH = os.path.join(os.path.dirname(__file__), "cuda", "libdwf.so")
_registered = False


def available():
    return os.path.exists(_SO_PATH)


def _ensureRegistered():
    global _registered
    if not _registered:
        lib = ctypes.CDLL(_SO_PATH)
        jax.ffi.register_ffi_target("dwf_apply_c64", jax.ffi.pycapsule(lib.DwfApplyC64),
                                    platform="CUDA")
        jax.ffi.register_ffi_target("dwf_apply_c128", jax.ffi.pycapsule(lib.DwfApplyC128),
                                    platform="CUDA")
        _registered = True


def applyD(settings, gaugeLinks, psi, sign):
    """Fused D_dwf (sign=+1) or D_dwf^dagger (sign=-1); drop-in for _applyD_xla.

    vmap adds a leading batch axis to gaugeLinks and psi (vmap_method
    'broadcast_all'); the kernel reads the batch size from the buffer ranks.
    """
    _ensureRegistered()
    target = "dwf_apply_c64" if psi.dtype == jnp.complex64 else "dwf_apply_c128"
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(psi.shape, psi.dtype),
                            vmap_method="broadcast_all")
    return call(gaugeLinks.astype(psi.dtype), psi,
                sign=np.int32(sign), a=np.float64(settings.a),
                M5=np.float64(settings.M5), fMass=np.float64(settings.fMass))
