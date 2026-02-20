#include <nrn/neuron/adex.h>

namespace nrn {
namespace neuron {

// Forward declaration of the CUDA kernel dispatch function.
namespace cuda {
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
    double dt);
} // namespace cuda

// ============================================================================
// Constructors
// ============================================================================

AdExImpl::AdExImpl(int64_t n)
    : AdExImpl(n, AdExOptions{}) {}

AdExImpl::AdExImpl(int64_t n, AdExOptions options)
    : options_(std::move(options)) {
    n_ = n;
    reset();
}

// ============================================================================
// reset()
// ============================================================================

void AdExImpl::reset() {
    auto opts = torch::TensorOptions().dtype(options_.dtype());

    if (!v.defined()) {
        // First call — register all buffers.
        v          = register_buffer("v",          torch::full({n_}, options_.v_rest(), opts));
        w          = register_buffer("w",          torch::zeros({n_}, opts));
        spike      = register_buffer("spike",      torch::zeros({n_}, opts));
        refractory = register_buffer("refractory", torch::zeros({n_}, opts));
        I_syn      = register_buffer("I_syn",      torch::zeros({n_}, opts));

        v_rest  = register_buffer("v_rest",  torch::full({n_}, options_.v_rest(),  opts));
        v_thresh = register_buffer("v_thresh", torch::full({n_}, options_.v_thresh(), opts));
        v_reset = register_buffer("v_reset", torch::full({n_}, options_.v_reset(), opts));
        v_peak  = register_buffer("v_peak",  torch::full({n_}, options_.v_peak(),  opts));
        tau_m   = register_buffer("tau_m",   torch::full({n_}, options_.tau_m(),   opts));
        tau_w   = register_buffer("tau_w",   torch::full({n_}, options_.tau_w(),   opts));
        tau_ref = register_buffer("tau_ref", torch::full({n_}, options_.tau_ref(), opts));
        c_m     = register_buffer("c_m",     torch::full({n_}, options_.c_m(),     opts));
        g_l     = register_buffer("g_l",     torch::full({n_}, options_.g_l(),     opts));
        a       = register_buffer("a",       torch::full({n_}, options_.a(),       opts));
        b       = register_buffer("b",       torch::full({n_}, options_.b(),       opts));
        delta_t = register_buffer("delta_t", torch::full({n_}, options_.delta_t(), opts));
        i_bg    = register_buffer("i_bg",    torch::full({n_}, options_.i_bg(),    opts));
    } else {
        // Subsequent calls — reinitialize state in-place.
        v.fill_(options_.v_rest());
        w.zero_();
        spike.zero_();
        refractory.zero_();
        I_syn.zero_();
    }
}

// ============================================================================
// forward()
// ============================================================================

void AdExImpl::forward(nrn::State& state, nrn::Time t, nrn::Duration dt) {
    // TODO: dispatch to CUDA kernel when tensors are on GPU.
    //
    // The AdEx dynamics are:
    //   c_m * dv/dt = -g_l*(v - v_rest)
    //                 + g_l*delta_t*exp((v - v_thresh)/delta_t)
    //                 - w + I_syn + i_bg
    //   tau_w * dw/dt = a*(v - v_rest) - w
    //
    //   if v >= v_peak:
    //       spike = 1, v = v_reset, w += b, refractory = tau_ref
    //
    // Will be implemented by cuda::adex_forward_cuda().
    //
    // Publish state into the State bag.
    state.set("v", v);
    state.set("w", w);
    state.set("spike", spike);
    state.set("refractory", refractory);
    state.set("I_syn", I_syn);
}

// ============================================================================
// state_vars()
// ============================================================================

std::vector<std::string> AdExImpl::state_vars() const {
    return {"v", "w", "spike", "refractory", "I_syn"};
}

} // namespace neuron
} // namespace nrn
