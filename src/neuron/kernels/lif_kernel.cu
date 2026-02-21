#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace nrn {
namespace neuron {
namespace cuda {

// ============================================================================
// LIF forward kernel
// ============================================================================
//
// Each thread handles one neuron. Forward Euler integration:
//
//   if refractory[i] > 0:
//       refractory[i] -= dt
//       spike[i] = 0
//   else:
//       dv = dt * (-(v[i] - v_rest[i]) / tau_m[i]
//            + (I_syn[i] + i_bg[i]) / c_m[i])
//       v[i] += dv
//       if v[i] >= v_thresh[i]:
//           spike[i] = 1
//           v[i] = v_reset[i]
//           refractory[i] = tau_ref[i]
//       else:
//           spike[i] = 0
//
//   I_syn[i] = 0   (consumed; will be refilled by synapse module next step)
// ============================================================================

template <typename scalar_t>
__global__ void lif_forward_kernel(
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ spike,
    scalar_t* __restrict__ refractory,
    scalar_t* __restrict__ I_syn,
    const scalar_t* __restrict__ v_rest,
    const scalar_t* __restrict__ v_thresh,
    const scalar_t* __restrict__ v_reset,
    const scalar_t* __restrict__ tau_m,
    const scalar_t* __restrict__ tau_ref,
    const scalar_t* __restrict__ c_m,
    const scalar_t* __restrict__ i_bg,
    const scalar_t dt,
    const int64_t N)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    scalar_t v_i = v[i];
    scalar_t ref_i = refractory[i];
    scalar_t spike_i = static_cast<scalar_t>(0);

    if (ref_i > static_cast<scalar_t>(0)) {
        // Still refractory — decrement timer, no voltage update.
        ref_i -= dt;
    } else {
        // Forward Euler integration of membrane potential.
        scalar_t dv = dt * (-(v_i - v_rest[i]) / tau_m[i]
                            + (I_syn[i] + i_bg[i]) / c_m[i]);
        v_i += dv;

        // Spike detection.
        if (v_i >= v_thresh[i]) {
            spike_i = static_cast<scalar_t>(1);
            v_i = v_reset[i];
            ref_i = tau_ref[i];
        }
    }

    // Write back.
    v[i] = v_i;
    spike[i] = spike_i;
    refractory[i] = ref_i;
    I_syn[i] = static_cast<scalar_t>(0);  // consumed
}

// ============================================================================
// Host dispatch function
// ============================================================================

void lif_forward_cuda(
    torch::Tensor v,
    torch::Tensor spike,
    torch::Tensor refractory,
    torch::Tensor I_syn,
    const torch::Tensor& v_rest,
    const torch::Tensor& v_thresh,
    const torch::Tensor& v_reset,
    const torch::Tensor& tau_m,
    const torch::Tensor& tau_ref,
    const torch::Tensor& c_m,
    const torch::Tensor& i_bg,
    double dt)
{
    const int64_t N = v.numel();
    TORCH_CHECK(N > 0, "lif_forward_cuda: neuron count must be > 0");
    TORCH_CHECK(v.is_cuda(), "lif_forward_cuda: tensors must be on CUDA device");

    const int threads = 256;
    const int blocks = (static_cast<int>(N) + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "lif_forward_cuda", [&] {
        lif_forward_kernel<scalar_t><<<blocks, threads>>>(
            v.data_ptr<scalar_t>(),
            spike.data_ptr<scalar_t>(),
            refractory.data_ptr<scalar_t>(),
            I_syn.data_ptr<scalar_t>(),
            v_rest.data_ptr<scalar_t>(),
            v_thresh.data_ptr<scalar_t>(),
            v_reset.data_ptr<scalar_t>(),
            tau_m.data_ptr<scalar_t>(),
            tau_ref.data_ptr<scalar_t>(),
            c_m.data_ptr<scalar_t>(),
            i_bg.data_ptr<scalar_t>(),
            static_cast<scalar_t>(dt),
            N);
    });

    // Check for kernel launch errors.
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cuda
} // namespace neuron
} // namespace nrn
