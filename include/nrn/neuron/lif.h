#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

// ============================================================================
// LIF — Leaky Integrate-and-Fire neuron model
// ============================================================================
//
// State variables (all shape [N]):
//   v          — membrane potential (V)
//   spike      — binary spike indicator (0 or 1)
//   refractory — remaining refractory time (s), 0 when not refractory
//   I_syn      — total synaptic input current (A)
//
// Dynamics (forward Euler):
//   if refractory > 0:
//       refractory -= dt
//   else:
//       dv/dt = (-(v - v_rest) / tau_m + (I_syn + i_bg) / c_m)
//   if v >= v_thresh:
//       spike = 1, v = v_reset, refractory = tau_ref
//
// Parameters stored as 1-D tensors [N] for per-neuron heterogeneity.
// ============================================================================

struct LIFNeuron {
    int64_t n;
    LIFOptions options;

    // State tensors [N]
    torch::Tensor v;
    torch::Tensor spike;
    torch::Tensor refractory;
    torch::Tensor I_syn;

    // Parameter tensors [N]
    torch::Tensor v_rest;
    torch::Tensor v_thresh;
    torch::Tensor v_reset;
    torch::Tensor tau_m;
    torch::Tensor tau_ref;
    torch::Tensor c_m;
    torch::Tensor i_bg;
};

// Lifecycle
LIFNeuron* lif_create(int64_t n, LIFOptions opts = {});
void lif_destroy(LIFNeuron* lif);

// Operations (also callable directly)
void lif_forward(void* self, State& state, double t, double dt);
void lif_reset(void* self);
const char** lif_state_vars(void* self, int* count);
int64_t lif_size(void* self);
void lif_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void lif_forward(LIFNeuron* lif, State& state, double t, double dt) {
    lif_forward(static_cast<void*>(lif), state, t, dt);
}
inline void lif_reset(LIFNeuron* lif) {
    lif_reset(static_cast<void*>(lif));
}
inline void lif_to_device(LIFNeuron* lif, torch::Device device) {
    lif_to_device(static_cast<void*>(lif), device);
}

// Ops table
extern nrn_ops lif_ops;

// Wrap as generic module handle
inline NrnModule lif_as_module(LIFNeuron* lif) {
    return NrnModule{static_cast<void*>(lif), &lif_ops};
}

} // namespace neuron
} // namespace nrn
