#include <nrn/neuron/lif.h>

namespace nrn {
namespace neuron {

// Forward declaration of the CUDA kernel dispatch function.
namespace cuda {
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
    double dt);
} // namespace cuda

// ============================================================================
// Constructors
// ============================================================================

LIFImpl::LIFImpl(int64_t n)
    : LIFImpl(n, LIFOptions{}) {}

LIFImpl::LIFImpl(int64_t n, LIFOptions options)
    : options_(std::move(options)) {
    n_ = n;
    reset();
}

// ============================================================================
// reset()
// ============================================================================

void LIFImpl::reset() {
    auto opts = torch::TensorOptions().dtype(options_.dtype());

    if (!v.defined()) {
        // First call — register all buffers.
        v          = register_buffer("v",          torch::full({n_}, options_.v_rest(), opts));
        spike      = register_buffer("spike",      torch::zeros({n_}, opts));
        refractory = register_buffer("refractory", torch::zeros({n_}, opts));
        I_syn      = register_buffer("I_syn",      torch::zeros({n_}, opts));

        v_rest  = register_buffer("v_rest",  torch::full({n_}, options_.v_rest(),  opts));
        v_thresh = register_buffer("v_thresh", torch::full({n_}, options_.v_thresh(), opts));
        v_reset = register_buffer("v_reset", torch::full({n_}, options_.v_reset(), opts));
        tau_m   = register_buffer("tau_m",   torch::full({n_}, options_.tau_m(),   opts));
        tau_ref = register_buffer("tau_ref", torch::full({n_}, options_.tau_ref(), opts));
        c_m     = register_buffer("c_m",     torch::full({n_}, options_.c_m(),     opts));
        i_bg    = register_buffer("i_bg",    torch::full({n_}, options_.i_bg(),    opts));
    } else {
        // Subsequent calls — reinitialize state in-place.
        v.fill_(options_.v_rest());
        spike.zero_();
        refractory.zero_();
        I_syn.zero_();
    }
}

// ============================================================================
// forward()
// ============================================================================

void LIFImpl::forward(nrn::State& state, nrn::Time t, nrn::Duration dt) {
    // TODO: dispatch to CUDA kernel when tensors are on GPU.
    //
    // For now this is a placeholder. The actual integration will be performed
    // by cuda::lif_forward_cuda() which launches the GPU kernel.
    //
    // CPU fallback (forward Euler) could be added here behind a device check:
    //
    //   if (v.is_cuda()) {
    //       cuda::lif_forward_cuda(v, spike, refractory, I_syn,
    //                              v_rest, v_thresh, v_reset,
    //                              tau_m, tau_ref, c_m, i_bg, dt);
    //   } else {
    //       // CPU path ...
    //   }
    //
    // Publish state tensors into the State bag so downstream modules
    // (synapses, monitors, etc.) can read them.
    state.set("v", v);
    state.set("spike", spike);
    state.set("refractory", refractory);
    state.set("I_syn", I_syn);
}

// ============================================================================
// state_vars()
// ============================================================================

std::vector<std::string> LIFImpl::state_vars() const {
    return {"v", "spike", "refractory", "I_syn"};
}

} // namespace neuron
} // namespace nrn
