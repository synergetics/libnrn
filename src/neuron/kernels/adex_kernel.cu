#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace nrn {
namespace neuron {
namespace cuda {

// ============================================================================
// AdEx forward kernel
// ============================================================================
//
// Each thread handles one neuron. Forward Euler integration:
//
//   if refractory[i] > 0:
//       refractory[i] -= dt
//       spike[i] = 0
//   else:
//       // Membrane potential dynamics
//       I_exp = g_l[i] * delta_t[i] * exp((v[i] - v_thresh[i]) / delta_t[i])
//       dv = dt * (-g_l[i]*(v[i] - v_rest[i]) + I_exp - w[i] + I_syn[i] + i_bg[i]) / c_m[i]
//       v[i] += dv
//
//       // Adaptation dynamics
//       dw = dt * (a[i]*(v[i] - v_rest[i]) - w[i]) / tau_w[i]
//       w[i] += dw
//
//       // Spike detection
//       if v[i] >= v_peak[i]:
//           spike[i] = 1
//           v[i] = v_reset[i]
//           w[i] += b[i]
//           refractory[i] = tau_ref[i]
//       else:
//           spike[i] = 0
//
//   I_syn[i] = 0   (consumed)
// ============================================================================

template <typename scalar_t>
__global__ void adex_forward_kernel(
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ w,
    scalar_t* __restrict__ spike,
    scalar_t* __restrict__ refractory,
    scalar_t* __restrict__ I_syn,
    const scalar_t* __restrict__ v_rest,
    const scalar_t* __restrict__ v_thresh,
    const scalar_t* __restrict__ v_reset,
    const scalar_t* __restrict__ v_peak,
    const scalar_t* __restrict__ tau_m,
    const scalar_t* __restrict__ tau_w,
    const scalar_t* __restrict__ tau_ref,
    const scalar_t* __restrict__ c_m,
    const scalar_t* __restrict__ g_l,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ delta_t,
    const scalar_t* __restrict__ i_bg,
    const scalar_t dt,
    const int64_t N)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    scalar_t v_i = v[i];
    scalar_t w_i = w[i];
    scalar_t ref_i = refractory[i];
    scalar_t spike_i = static_cast<scalar_t>(0);

    if (ref_i > static_cast<scalar_t>(0)) {
        // Still refractory — decrement timer, no voltage update.
        ref_i -= dt;
    } else {
        // Exponential spike-initiation current.
        scalar_t I_exp = g_l[i] * delta_t[i]
            * static_cast<scalar_t>(
                exp(static_cast<double>((v_i - v_thresh[i]) / delta_t[i])));

        // Forward Euler: membrane potential.
        scalar_t dv = dt * (-g_l[i] * (v_i - v_rest[i]) + I_exp
                            - w_i + I_syn[i] + i_bg[i]) / c_m[i];
        v_i += dv;
    }

    // Adaptation dynamics — evolves even during refractory (Brette & Gerstner 2005).
    scalar_t dw = dt * (a[i] * (v_i - v_rest[i]) - w_i) / tau_w[i];
    w_i += dw;

    // Spike detection (only if not refractory).
    if (ref_i <= static_cast<scalar_t>(0) && v_i >= v_peak[i]) {
        spike_i = static_cast<scalar_t>(1);
        v_i = v_reset[i];
        w_i += b[i];
        ref_i = tau_ref[i];
    }

    // Write back.
    v[i] = v_i;
    w[i] = w_i;
    spike[i] = spike_i;
    refractory[i] = ref_i;
    I_syn[i] = static_cast<scalar_t>(0);  // consumed
}

// ============================================================================
// Host dispatch function
// ============================================================================

void adex_forward_cuda(
    torch::Tensor v,
    torch::Tensor w,
    torch::Tensor spike,
    torch::Tensor refractory,
    torch::Tensor I_syn,
    const torch::Tensor& v_rest,
    const torch::Tensor& v_thresh,
    const torch::Tensor& v_reset,
    const torch::Tensor& v_peak,
    const torch::Tensor& tau_m,
    const torch::Tensor& tau_w,
    const torch::Tensor& tau_ref,
    const torch::Tensor& c_m,
    const torch::Tensor& g_l,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& delta_t,
    const torch::Tensor& i_bg,
    double dt)
{
    const int64_t N = v.numel();
    TORCH_CHECK(N > 0, "adex_forward_cuda: neuron count must be > 0");
    TORCH_CHECK(v.is_cuda(), "adex_forward_cuda: tensors must be on CUDA device");

    const int threads = 256;
    const int blocks = (static_cast<int>(N) + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "adex_forward_cuda", [&] {
        adex_forward_kernel<scalar_t><<<blocks, threads>>>(
            v.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            spike.data_ptr<scalar_t>(),
            refractory.data_ptr<scalar_t>(),
            I_syn.data_ptr<scalar_t>(),
            v_rest.data_ptr<scalar_t>(),
            v_thresh.data_ptr<scalar_t>(),
            v_reset.data_ptr<scalar_t>(),
            v_peak.data_ptr<scalar_t>(),
            tau_m.data_ptr<scalar_t>(),
            tau_w.data_ptr<scalar_t>(),
            tau_ref.data_ptr<scalar_t>(),
            c_m.data_ptr<scalar_t>(),
            g_l.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            delta_t.data_ptr<scalar_t>(),
            i_bg.data_ptr<scalar_t>(),
            static_cast<scalar_t>(dt),
            N);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cuda
} // namespace neuron
} // namespace nrn
