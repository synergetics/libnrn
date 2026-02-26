#include <nrn/neuron/izhikevich.h>

namespace nrn {
namespace neuron {

// CUDA kernel dispatch (defined in kernels/izhikevich_kernel.cu).
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

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_ops izh_ops = {
    .forward   = izh_forward,
    .reset     = izh_reset,
    .state_vars = izh_state_vars,
    .size      = izh_size,
    .to_device = izh_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

IzhikevichNeuron* izh_create(int64_t n, IzhikevichOptions opts) {
    auto* izh = new IzhikevichNeuron();
    izh->n = n;
    izh->options = std::move(opts);

    auto topts = torch::TensorOptions().dtype(izh->options.dtype());

    // State tensors
    izh->v     = torch::full({n}, izh->options.v_init(), topts);
    izh->u     = torch::full({n}, izh->options.b() * izh->options.v_init(), topts);
    izh->spike = torch::zeros({n}, topts);
    izh->I_syn = torch::zeros({n}, topts);

    // Parameter tensors
    izh->a      = torch::full({n}, izh->options.a(),      topts);
    izh->b      = torch::full({n}, izh->options.b(),      topts);
    izh->c      = torch::full({n}, izh->options.c(),      topts);
    izh->d      = torch::full({n}, izh->options.d(),      topts);
    izh->v_peak = torch::full({n}, izh->options.v_peak(), topts);

    return izh;
}

void izh_destroy(IzhikevichNeuron* izh) {
    delete izh;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void izh_forward(void* self, State& state, double /*t*/, double dt) {
    auto* izh = static_cast<IzhikevichNeuron*>(self);

    if (izh->v.is_cuda()) {
        cuda::izhikevich_forward_cuda(izh->v, izh->u, izh->spike, izh->I_syn,
                                      izh->a, izh->b, izh->c, izh->d,
                                      izh->v_peak, dt);
    } else {
        // CPU path: vectorized tensor ops.
        double dt_ms = dt * 1000.0;

        // Two half-steps for v (Izhikevich's recommended scheme).
        auto dv1 = 0.04 * izh->v * izh->v + 5.0 * izh->v + 140.0 - izh->u + izh->I_syn;
        izh->v = izh->v + 0.5 * dt_ms * dv1;

        auto dv2 = 0.04 * izh->v * izh->v + 5.0 * izh->v + 140.0 - izh->u + izh->I_syn;
        izh->v = izh->v + 0.5 * dt_ms * dv2;

        // Recovery variable.
        auto du = dt_ms * izh->a * (izh->b * izh->v - izh->u);
        izh->u = izh->u + du;

        // Spike detection.
        auto spiked = (izh->v >= izh->v_peak);

        izh->v = torch::where(spiked, izh->c, izh->v);
        izh->u = torch::where(spiked, izh->u + izh->d, izh->u);
        izh->spike = spiked.to(izh->v.dtype());

        izh->I_syn.zero_();
    }

    state_set(state, "v", izh->v);
    state_set(state, "u", izh->u);
    state_set(state, "spike", izh->spike);
    state_set(state, "I_syn", izh->I_syn);
}

void izh_reset(void* self) {
    auto* izh = static_cast<IzhikevichNeuron*>(self);
    izh->v.fill_(izh->options.v_init());
    izh->u.fill_(izh->options.b() * izh->options.v_init());
    izh->spike.zero_();
    izh->I_syn.zero_();
}

static const char* izh_var_names[] = {"v", "u", "spike", "I_syn"};

const char** izh_state_vars(void* /*self*/, int* count) {
    *count = 4;
    return izh_var_names;
}

int64_t izh_size(void* self) {
    return static_cast<IzhikevichNeuron*>(self)->n;
}

void izh_to_device(void* self, torch::Device device) {
    auto* izh = static_cast<IzhikevichNeuron*>(self);
    izh->v      = izh->v.to(device);
    izh->u      = izh->u.to(device);
    izh->spike  = izh->spike.to(device);
    izh->I_syn  = izh->I_syn.to(device);
    izh->a      = izh->a.to(device);
    izh->b      = izh->b.to(device);
    izh->c      = izh->c.to(device);
    izh->d      = izh->d.to(device);
    izh->v_peak = izh->v_peak.to(device);
}

} // namespace neuron
} // namespace nrn
