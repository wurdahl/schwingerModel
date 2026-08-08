// Fused domain-wall Dirac operator for the 2D Schwinger model, as a JAX FFI
// custom call. One thread per (s5, x, t) site computes all nine stencil terms
// for both spin components in registers: one global read per neighbor, one
// write, no intermediates in DRAM (the XLA lowering of the same operator runs
// ~25 separate kernels).
//
// Equivalent to hmc_gpu._applyD_xla: sign=+1 is D, sign=-1 is D^dagger (the
// gammas are Hermitian, so the dagger is the same stencil with every gamma
// negated). Conventions (gamma_x = sigma_x, gamma_t = sigma_y, gamma_5 =
// sigma_z, anti-periodic time BC, fifth-dimension mass wrap) must match
// params.py / buildOps.buildDwfOp — verified against the sparse oracle.
//
// Build: ./build.sh (nvcc from the nvidia-cuda-nvcc pip wheel; sm_120)

#include <cuda_runtime.h>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <typename T>
struct Cx {
  T re, im;
  __device__ Cx() {}
  __device__ Cx(T r, T i) : re(r), im(i) {}
  __device__ Cx operator+(const Cx& o) const { return Cx(re + o.re, im + o.im); }
  __device__ Cx operator-(const Cx& o) const { return Cx(re - o.re, im - o.im); }
  __device__ Cx operator*(const Cx& o) const {
    return Cx(re * o.re - im * o.im, re * o.im + im * o.re);
  }
  __device__ Cx operator*(T s) const { return Cx(re * s, im * s); }
  __device__ Cx conj() const { return Cx(re, -im); }
  __device__ Cx timesI() const { return Cx(-im, re); }  // i*z
};

// U: (dimx, dimt, 2) with [..,0]=U_t, [..,1]=U_x;  psi/out: (dim5, dimx, dimt, 2 spin)
// A leading batch dimension on both (vmapped chains) is handled via strides.
template <typename T>
__global__ void dwfApply(const Cx<T>* __restrict__ U,
                         const Cx<T>* __restrict__ psi,
                         Cx<T>* __restrict__ out,
                         int S, int X, int TT, long nSites, long uBatchStride,
                         T c, T diag, T mass, T sgn) {
  long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (tid >= nSites) return;

  int t = tid % TT;
  long r = tid / TT;
  int x = r % X;  r /= X;
  int s = r % S;
  long b = r / S;

  const Cx<T>* Ub = U + b * uBatchStride;
  const Cx<T>* Pb = psi + b * (long)S * X * TT * 2;
  Cx<T>* Ob = out + b * (long)S * X * TT * 2;

  auto P = [&](int ss, int xx, int tt, int al) -> Cx<T> {
    return Pb[(((long)ss * X + xx) * TT + tt) * 2 + al];
  };
  auto L = [&](int xx, int tt, int mu) -> Cx<T> {
    return Ub[((long)xx * TT + tt) * 2 + mu];
  };

  int xp = (x + 1 == X) ? 0 : x + 1;
  int xm = (x == 0) ? X - 1 : x - 1;
  int tp = (t + 1 == TT) ? 0 : t + 1;
  int tm = (t == 0) ? TT - 1 : t - 1;
  int sp = (s + 1 == S) ? 0 : s + 1;
  int sm = (s == 0) ? S - 1 : s - 1;

  // anti-periodic time BC: sign flip where the hop wraps around
  T bcf = (t == TT - 1) ? (T)-1 : (T)1;
  T bcb = (t == 0) ? (T)-1 : (T)1;
  // fifth dimension: the wrap-around hop carries -fMass (the mass term)
  T mf = (s == S - 1) ? -mass : (T)1;
  T mb = (s == 0) ? -mass : (T)1;

  Cx<T> o0 = P(s, x, t, 0) * diag;
  Cx<T> o1 = P(s, x, t, 1) * diag;

  {  // +x hop: -c * U_x(x,t) * (1 - sgn*gx) psi(x+1) ; (1-s*gx)psi = (p0-s*p1, p1-s*p0)
    Cx<T> q0 = P(s, xp, t, 0), q1 = P(s, xp, t, 1);
    Cx<T> u = L(x, t, 1) * c;
    o0 = o0 - u * (q0 - q1 * sgn);
    o1 = o1 - u * (q1 - q0 * sgn);
  }
  {  // -x hop: -c * conj(U_x(x-1,t)) * (1 + sgn*gx) psi(x-1)
    Cx<T> q0 = P(s, xm, t, 0), q1 = P(s, xm, t, 1);
    Cx<T> u = L(xm, t, 1).conj() * c;
    o0 = o0 - u * (q0 + q1 * sgn);
    o1 = o1 - u * (q1 + q0 * sgn);
  }
  {  // +t hop: -c*bcf * U_t(x,t) * (1 - sgn*gt) psi(t+1) ; (1-s*gt)psi = (p0+i s p1, p1-i s p0)
    Cx<T> q0 = P(s, x, tp, 0), q1 = P(s, x, tp, 1);
    Cx<T> u = L(x, t, 0) * (c * bcf);
    o0 = o0 - u * (q0 + q1.timesI() * sgn);
    o1 = o1 - u * (q1 - q0.timesI() * sgn);
  }
  {  // -t hop: -c*bcb * conj(U_t(x,t-1)) * (1 + sgn*gt) psi(t-1)
    Cx<T> q0 = P(s, x, tm, 0), q1 = P(s, x, tm, 1);
    Cx<T> u = L(x, tm, 0).conj() * (c * bcb);
    o0 = o0 - u * (q0 - q1.timesI() * sgn);
    o1 = o1 - u * (q1 + q0.timesI() * sgn);
  }

  // fifth-dimension hops through the chiral projectors, diagonal in spin:
  // forward projector (1 - sgn*g5)/2 = diag(0,1) for D, diag(1,0) for D^dagger
  if (sgn > (T)0) {
    o1 = o1 - P(sp, x, t, 1) * mf;
    o0 = o0 - P(sm, x, t, 0) * mb;
  } else {
    o0 = o0 - P(sp, x, t, 0) * mf;
    o1 = o1 - P(sm, x, t, 1) * mb;
  }

  Ob[(((long)s * X + x) * TT + t) * 2 + 0] = o0;
  Ob[(((long)s * X + x) * TT + t) * 2 + 1] = o1;
}

template <ffi::DataType DT>
ffi::Error dwfImpl(cudaStream_t stream, ffi::Buffer<DT> U, ffi::Buffer<DT> psi,
                   ffi::Result<ffi::Buffer<DT>> out, int32_t sign, double a,
                   double M5, double fMass) {
  using NT = typename ffi::NativeType<DT>;  // std::complex<float/double>
  using T = typename NT::value_type;

  auto dims = psi.dimensions();
  int rank = dims.size();
  if (rank != 4 && rank != 5)
    return ffi::Error::InvalidArgument("psi must be rank 4 (dim5,x,t,2) or 5 (batched)");

  long B = (rank == 5) ? dims[0] : 1;
  int S = dims[rank - 4], X = dims[rank - 3], TT = dims[rank - 2];
  long uBatchStride = (U.dimensions().size() == 4) ? (long)X * TT * 2 : 0;

  long nSites = B * (long)S * X * TT;
  int block = 256;
  long grid = (nSites + block - 1) / block;

  dwfApply<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const Cx<T>*>(U.typed_data()),
      reinterpret_cast<const Cx<T>*>(psi.typed_data()),
      reinterpret_cast<Cx<T>*>(out->typed_data()), S, X, TT, nSites,
      uBatchStride, (T)(1.0 / (2.0 * a)), (T)(1.0 - M5 + 2.0 / a), (T)fMass,
      (T)sign);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DwfApplyC64, (dwfImpl<ffi::DataType::C64>),
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::C64>>()
                                  .Arg<ffi::Buffer<ffi::DataType::C64>>()
                                  .Ret<ffi::Buffer<ffi::DataType::C64>>()
                                  .Attr<int32_t>("sign")
                                  .Attr<double>("a")
                                  .Attr<double>("M5")
                                  .Attr<double>("fMass"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DwfApplyC128, (dwfImpl<ffi::DataType::C128>),
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>()
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>()
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>()
                                  .Attr<int32_t>("sign")
                                  .Attr<double>("a")
                                  .Attr<double>("M5")
                                  .Attr<double>("fMass"));
