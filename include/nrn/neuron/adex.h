#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

// ============================================================================
// AdEx — Adaptive Exponential Integrate-and-Fire neuron model
// ============================================================================
//
// State variables (all shape [N]):
//   v          — membrane potential (V)
//   w          — adaptation current (A)
//   spike      — binary spike indicator (0 or 1)
//   refractory — remaining refractory time (s)
//   I_syn      — total synaptic input current (A)
//
// Dynamics (Brette & Gerstner 2005):
//   c_m * dv/dt = -g_l*(v - v_rest) + g_l*delta_t*exp((v - v_thresh)/delta_t)
//                 - w + I_syn + i_bg
//   tau_w * dw/dt = a*(v - v_rest) - w
//
//   if v >= v_peak:
//       spike = 1, v = v_reset, w += b, refractory = tau_ref
//
// Parameters stored as 1-D tensors [N] for per-neuron heterogeneity.
// ============================================================================

struct AdExNeuron {
    int64_t n;
    AdExOptions options;

    // State tensors [N]
    torch::Tensor v;
    torch::Tensor w;
    torch::Tensor spike;
    torch::Tensor refractory;
    torch::Tensor I_syn;

    // Parameter tensors [N]
    torch::Tensor v_rest;
    torch::Tensor v_thresh;
    torch::Tensor v_reset;
    torch::Tensor v_peak;
    torch::Tensor tau_m;
    torch::Tensor tau_w;
    torch::Tensor tau_ref;
    torch::Tensor c_m;
    torch::Tensor g_l;
    torch::Tensor a;
    torch::Tensor b;
    torch::Tensor delta_t;
    torch::Tensor i_bg;
};

// Lifecycle
AdExNeuron* adex_create(int64_t n, AdExOptions opts = {});
void adex_destroy(AdExNeuron* adex);

// Operations
void adex_forward(void* self, State& state, double t, double dt);
void adex_reset(void* self);
const char** adex_state_vars(void* self, int* count);
int64_t adex_size(void* self);
void adex_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void adex_forward(AdExNeuron* adex, State& state, double t, double dt) {
    adex_forward(static_cast<void*>(adex), state, t, dt);
}
inline void adex_reset(AdExNeuron* adex) {
    adex_reset(static_cast<void*>(adex));
}
inline void adex_to_device(AdExNeuron* adex, torch::Device device) {
    adex_to_device(static_cast<void*>(adex), device);
}

// Ops table
extern nrn_ops adex_ops;

// Wrap as generic module handle
inline NrnModule adex_as_module(AdExNeuron* adex) {
    return NrnModule{static_cast<void*>(adex), &adex_ops};
}

} // namespace neuron
} // namespace nrn
