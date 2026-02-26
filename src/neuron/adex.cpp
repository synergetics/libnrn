#include <nrn/neuron/adex.h>

namespace nrn {
namespace neuron {

// CUDA kernel dispatch (defined in kernels/adex_kernel.cu).
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

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_ops adex_ops = {
    .forward   = adex_forward,
    .reset     = adex_reset,
    .state_vars = adex_state_vars,
    .size      = adex_size,
    .to_device = adex_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

AdExNeuron* adex_create(int64_t n, AdExOptions opts) {
    auto* adex = new AdExNeuron();
    adex->n = n;
    adex->options = std::move(opts);

    auto topts = torch::TensorOptions().dtype(adex->options.dtype());

    // State tensors
    adex->v          = torch::full({n}, adex->options.v_rest(), topts);
    adex->w          = torch::zeros({n}, topts);
    adex->spike      = torch::zeros({n}, topts);
    adex->refractory = torch::zeros({n}, topts);
    adex->I_syn      = torch::zeros({n}, topts);

    // Parameter tensors
    adex->v_rest  = torch::full({n}, adex->options.v_rest(),  topts);
    adex->v_thresh = torch::full({n}, adex->options.v_thresh(), topts);
    adex->v_reset = torch::full({n}, adex->options.v_reset(), topts);
    adex->v_peak  = torch::full({n}, adex->options.v_peak(),  topts);
    adex->tau_m   = torch::full({n}, adex->options.tau_m(),   topts);
    adex->tau_w   = torch::full({n}, adex->options.tau_w(),   topts);
    adex->tau_ref = torch::full({n}, adex->options.tau_ref(), topts);
    adex->c_m     = torch::full({n}, adex->options.c_m(),     topts);
    adex->g_l     = torch::full({n}, adex->options.g_l(),     topts);
    adex->a       = torch::full({n}, adex->options.a(),       topts);
    adex->b       = torch::full({n}, adex->options.b(),       topts);
    adex->delta_t = torch::full({n}, adex->options.delta_t(), topts);
    adex->i_bg    = torch::full({n}, adex->options.i_bg(),    topts);

    return adex;
}

void adex_destroy(AdExNeuron* adex) {
    delete adex;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void adex_forward(void* self, State& state, double /*t*/, double dt) {
    auto* adex = static_cast<AdExNeuron*>(self);

    if (adex->v.is_cuda()) {
        cuda::adex_forward_cuda(adex->v, adex->w, adex->spike, adex->refractory,
                                adex->I_syn, adex->v_rest, adex->v_thresh,
                                adex->v_reset, adex->v_peak, adex->tau_m,
                                adex->tau_w, adex->tau_ref, adex->c_m,
                                adex->g_l, adex->a, adex->b, adex->delta_t,
                                adex->i_bg, dt);
    } else {
        // CPU path: vectorized tensor ops.
        auto active = (adex->refractory <= 0);

        adex->refractory = torch::where(adex->refractory > 0,
                                        adex->refractory - dt,
                                        adex->refractory);

        // Exponential spike-initiation current.
        auto I_exp = adex->g_l * adex->delta_t
                     * torch::exp((adex->v - adex->v_thresh) / adex->delta_t);

        auto dv = dt * (-adex->g_l * (adex->v - adex->v_rest) + I_exp
                        - adex->w + adex->I_syn + adex->i_bg) / adex->c_m;
        adex->v = torch::where(active, adex->v + dv, adex->v);

        // Adaptation — evolves even during refractory.
        auto dw = dt * (adex->a * (adex->v - adex->v_rest) - adex->w) / adex->tau_w;
        adex->w = adex->w + dw;

        auto spiked = (adex->v >= adex->v_peak) & active;

        adex->v = torch::where(spiked, adex->v_reset, adex->v);
        adex->w = torch::where(spiked, adex->w + adex->b, adex->w);
        adex->refractory = torch::where(spiked, adex->tau_ref, adex->refractory);
        adex->spike = spiked.to(adex->v.dtype());

        adex->I_syn.zero_();
    }

    state_set(state, "v", adex->v);
    state_set(state, "w", adex->w);
    state_set(state, "spike", adex->spike);
    state_set(state, "refractory", adex->refractory);
    state_set(state, "I_syn", adex->I_syn);
}

void adex_reset(void* self) {
    auto* adex = static_cast<AdExNeuron*>(self);
    adex->v.fill_(adex->options.v_rest());
    adex->w.zero_();
    adex->spike.zero_();
    adex->refractory.zero_();
    adex->I_syn.zero_();
}

static const char* adex_var_names[] = {"v", "w", "spike", "refractory", "I_syn"};

const char** adex_state_vars(void* /*self*/, int* count) {
    *count = 5;
    return adex_var_names;
}

int64_t adex_size(void* self) {
    return static_cast<AdExNeuron*>(self)->n;
}

void adex_to_device(void* self, torch::Device device) {
    auto* adex = static_cast<AdExNeuron*>(self);
    adex->v          = adex->v.to(device);
    adex->w          = adex->w.to(device);
    adex->spike      = adex->spike.to(device);
    adex->refractory = adex->refractory.to(device);
    adex->I_syn      = adex->I_syn.to(device);
    adex->v_rest     = adex->v_rest.to(device);
    adex->v_thresh   = adex->v_thresh.to(device);
    adex->v_reset    = adex->v_reset.to(device);
    adex->v_peak     = adex->v_peak.to(device);
    adex->tau_m      = adex->tau_m.to(device);
    adex->tau_w      = adex->tau_w.to(device);
    adex->tau_ref    = adex->tau_ref.to(device);
    adex->c_m        = adex->c_m.to(device);
    adex->g_l        = adex->g_l.to(device);
    adex->a          = adex->a.to(device);
    adex->b          = adex->b.to(device);
    adex->delta_t    = adex->delta_t.to(device);
    adex->i_bg       = adex->i_bg.to(device);
}

} // namespace neuron
} // namespace nrn
