#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace nrn {
namespace neuron {
namespace cuda {

// ============================================================================
// Izhikevich forward kernel (dimensionless convention)
// ============================================================================
//
// Each thread handles one neuron. Forward Euler integration:
//
//   dt_ms = dt * 1000.0   (convert SI seconds to model ms)
//
//   // Two half-steps for v (improved numerical stability per Izhikevich's
//   // recommendation):
//   v[i] += 0.5 * dt_ms * (0.04*v[i]*v[i] + 5*v[i] + 140 - u[i] + I_syn[i])
//   v[i] += 0.5 * dt_ms * (0.04*v[i]*v[i] + 5*v[i] + 140 - u[i] + I_syn[i])
//
//   // Recovery variable
//   du = dt_ms * a[i] * (b[i]*v[i] - u[i])
//   u[i] += du
//
//   // Spike detection
//   if v[i] >= v_peak[i]:
//       spike[i] = 1
//       v[i] = c[i]
//       u[i] += d[i]
//   else:
//       spike[i] = 0
//
//   I_syn[i] = 0   (consumed)
// ============================================================================

template <typename scalar_t>
__global__ void izhikevich_forward_kernel(
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ spike,
    scalar_t* __restrict__ I_syn,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ d,
    const scalar_t* __restrict__ v_peak,
    const scalar_t dt,
    const int64_t N)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // TODO: implement Izhikevich dynamics
    // This kernel is a scaffold — the integration logic will be filled in
    // during Phase 1 development. The structure (grid, block, data pointers)
    // is established here so the build system and dispatch path work end-to-end.
}

// ============================================================================
// Host dispatch function
// ============================================================================

void izhikevich_forward_cuda(
    torch::Tensor v,
    torch::Tensor u,
    torch::Tensor spike,
    torch::Tensor I_syn,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& d,
    const torch::Tensor& v_peak,
    double dt)
{
    const int64_t N = v.numel();
    TORCH_CHECK(N > 0, "izhikevich_forward_cuda: neuron count must be > 0");
    TORCH_CHECK(v.is_cuda(), "izhikevich_forward_cuda: tensors must be on CUDA device");

    const int threads = 256;
    const int blocks = (static_cast<int>(N) + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(v.scalar_type(), "izhikevich_forward_cuda", [&] {
        izhikevich_forward_kernel<scalar_t><<<blocks, threads>>>(
            v.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            spike.data_ptr<scalar_t>(),
            I_syn.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            v_peak.data_ptr<scalar_t>(),
            static_cast<scalar_t>(dt),
            N);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cuda
} // namespace neuron
} // namespace nrn
