#include <nrn/neuron/lif.h>

namespace nrn {
namespace neuron {

// CUDA kernel dispatch (defined in kernels/lif_kernel.cu).
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

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_ops lif_ops = {
    .forward   = lif_forward,
    .reset     = lif_reset,
    .state_vars = lif_state_vars,
    .size      = lif_size,
    .to_device = lif_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

LIFNeuron* lif_create(int64_t n, LIFOptions opts) {
    auto* lif = new LIFNeuron();
    lif->n = n;
    lif->options = std::move(opts);

    auto topts = torch::TensorOptions().dtype(lif->options.dtype());

    // State tensors
    lif->v          = torch::full({n}, lif->options.v_rest(), topts);
    lif->spike      = torch::zeros({n}, topts);
    lif->refractory = torch::zeros({n}, topts);
    lif->I_syn      = torch::zeros({n}, topts);

    // Parameter tensors
    lif->v_rest  = torch::full({n}, lif->options.v_rest(),  topts);
    lif->v_thresh = torch::full({n}, lif->options.v_thresh(), topts);
    lif->v_reset = torch::full({n}, lif->options.v_reset(), topts);
    lif->tau_m   = torch::full({n}, lif->options.tau_m(),   topts);
    lif->tau_ref = torch::full({n}, lif->options.tau_ref(), topts);
    lif->c_m     = torch::full({n}, lif->options.c_m(),     topts);
    lif->i_bg    = torch::full({n}, lif->options.i_bg(),    topts);

    return lif;
}

void lif_destroy(LIFNeuron* lif) {
    delete lif;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void lif_forward(void* self, State& state, double /*t*/, double dt) {
    auto* lif = static_cast<LIFNeuron*>(self);

    if (lif->v.is_cuda()) {
        cuda::lif_forward_cuda(lif->v, lif->spike, lif->refractory, lif->I_syn,
                               lif->v_rest, lif->v_thresh, lif->v_reset,
                               lif->tau_m, lif->tau_ref, lif->c_m, lif->i_bg, dt);
    } else {
        // CPU path: vectorized tensor ops.
        auto active = (lif->refractory <= 0);

        lif->refractory = torch::where(lif->refractory > 0,
                                       lif->refractory - dt,
                                       lif->refractory);

        auto dv = dt * (-(lif->v - lif->v_rest) / lif->tau_m
                        + (lif->I_syn + lif->i_bg) / lif->c_m);
        lif->v = torch::where(active, lif->v + dv, lif->v);

        auto spiked = (lif->v >= lif->v_thresh) & active;

        lif->v = torch::where(spiked, lif->v_reset, lif->v);
        lif->refractory = torch::where(spiked, lif->tau_ref, lif->refractory);
        lif->spike = spiked.to(lif->v.dtype());

        lif->I_syn.zero_();
    }

    // Publish state.
    state_set(state, "v", lif->v);
    state_set(state, "spike", lif->spike);
    state_set(state, "refractory", lif->refractory);
    state_set(state, "I_syn", lif->I_syn);
}

void lif_reset(void* self) {
    auto* lif = static_cast<LIFNeuron*>(self);
    lif->v.fill_(lif->options.v_rest());
    lif->spike.zero_();
    lif->refractory.zero_();
    lif->I_syn.zero_();
}

static const char* lif_var_names[] = {"v", "spike", "refractory", "I_syn"};

const char** lif_state_vars(void* /*self*/, int* count) {
    *count = 4;
    return lif_var_names;
}

int64_t lif_size(void* self) {
    return static_cast<LIFNeuron*>(self)->n;
}

void lif_to_device(void* self, torch::Device device) {
    auto* lif = static_cast<LIFNeuron*>(self);
    lif->v          = lif->v.to(device);
    lif->spike      = lif->spike.to(device);
    lif->refractory = lif->refractory.to(device);
    lif->I_syn      = lif->I_syn.to(device);
    lif->v_rest     = lif->v_rest.to(device);
    lif->v_thresh   = lif->v_thresh.to(device);
    lif->v_reset    = lif->v_reset.to(device);
    lif->tau_m      = lif->tau_m.to(device);
    lif->tau_ref    = lif->tau_ref.to(device);
    lif->c_m        = lif->c_m.to(device);
    lif->i_bg       = lif->i_bg.to(device);
}

} // namespace neuron
} // namespace nrn
