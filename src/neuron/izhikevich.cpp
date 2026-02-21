#include <nrn/neuron/izhikevich.h>

namespace nrn {
namespace neuron {

// Forward declaration of the CUDA kernel dispatch function.
namespace cuda {
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
    double dt);
} // namespace cuda

// ============================================================================
// Constructors
// ============================================================================

IzhikevichImpl::IzhikevichImpl(int64_t n)
    : IzhikevichImpl(n, IzhikevichOptions{}) {}

IzhikevichImpl::IzhikevichImpl(int64_t n, IzhikevichOptions options)
    : options_(std::move(options)) {
    n_ = n;
    reset();
}

// ============================================================================
// reset()
// ============================================================================

void IzhikevichImpl::reset() {
    auto opts = torch::TensorOptions().dtype(options_.dtype());

    if (!v.defined()) {
        // First call — register all buffers.
        v     = register_buffer("v",     torch::full({n_}, options_.v_init(), opts));
        u     = register_buffer("u",     torch::full({n_}, options_.b() * options_.v_init(), opts));
        spike = register_buffer("spike", torch::zeros({n_}, opts));
        I_syn = register_buffer("I_syn", torch::zeros({n_}, opts));

        a      = register_buffer("a",      torch::full({n_}, options_.a(),      opts));
        b      = register_buffer("b",      torch::full({n_}, options_.b(),      opts));
        c      = register_buffer("c",      torch::full({n_}, options_.c(),      opts));
        d      = register_buffer("d",      torch::full({n_}, options_.d(),      opts));
        v_peak = register_buffer("v_peak", torch::full({n_}, options_.v_peak(), opts));
    } else {
        // Subsequent calls — reinitialize state in-place.
        v.fill_(options_.v_init());
        u.fill_(options_.b() * options_.v_init());
        spike.zero_();
        I_syn.zero_();
    }
}

// ============================================================================
// forward()
// ============================================================================

void IzhikevichImpl::forward(nrn::State& state, nrn::Time /*t*/, nrn::Duration dt) {
    if (v.is_cuda()) {
        cuda::izhikevich_forward_cuda(v, u, spike, I_syn,
                                      a, b, c, d, v_peak, dt);
    } else {
        // CPU path: vectorized PyTorch tensor operations.
        // Convert dt from SI seconds to model milliseconds.
        double dt_ms = dt * 1000.0;

        // Two half-steps for v (Izhikevich's recommended numerical scheme).
        auto dv1 = 0.04 * v * v + 5.0 * v + 140.0 - u + I_syn;
        v = v + 0.5 * dt_ms * dv1;

        auto dv2 = 0.04 * v * v + 5.0 * v + 140.0 - u + I_syn;
        v = v + 0.5 * dt_ms * dv2;

        // Recovery variable.
        auto du = dt_ms * a * (b * v - u);
        u = u + du;

        // Spike detection.
        auto spiked = (v >= v_peak);

        // Apply reset.
        v = torch::where(spiked, c, v);
        u = torch::where(spiked, u + d, u);
        spike = spiked.to(v.dtype());

        // Consume synaptic current.
        I_syn.zero_();
    }

    // Publish state into the State bag.
    state.set("v", v);
    state.set("u", u);
    state.set("spike", spike);
    state.set("I_syn", I_syn);
}

// ============================================================================
// state_vars()
// ============================================================================

std::vector<std::string> IzhikevichImpl::state_vars() const {
    return {"v", "u", "spike", "I_syn"};
}

} // namespace neuron
} // namespace nrn
