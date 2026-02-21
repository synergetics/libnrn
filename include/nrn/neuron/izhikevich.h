#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

// ============================================================================
// Izhikevich neuron model (dimensionless convention)
// ============================================================================
//
// State variables (all shape [N]):
//   v     — membrane potential (dimensionless, mV-like)
//   u     — recovery variable  (dimensionless)
//   spike — binary spike indicator (0 or 1)
//   I_syn — total synaptic input current (dimensionless)
//
// Dynamics (Izhikevich 2003):
//   dv/dt = 0.04*v^2 + 5*v + 140 - u + I_syn
//   du/dt = a * (b*v - u)
//
//   if v >= v_peak:
//       spike = 1, v = c, u += d
//
// Note: dt in forward() is seconds (SI); converted to ms internally.
// Parameters stored as 1-D tensors [N] for per-neuron heterogeneity.
// ============================================================================

struct IzhikevichNeuron {
    int64_t n;
    IzhikevichOptions options;

    // State tensors [N]
    torch::Tensor v;
    torch::Tensor u;
    torch::Tensor spike;
    torch::Tensor I_syn;

    // Parameter tensors [N]
    torch::Tensor a;
    torch::Tensor b;
    torch::Tensor c;
    torch::Tensor d;
    torch::Tensor v_peak;
};

// Lifecycle
IzhikevichNeuron* izh_create(int64_t n, IzhikevichOptions opts = {});
void izh_destroy(IzhikevichNeuron* izh);

// Operations
void izh_forward(void* self, State& state, double t, double dt);
void izh_reset(void* self);
const char** izh_state_vars(void* self, int* count);
int64_t izh_size(void* self);
void izh_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void izh_forward(IzhikevichNeuron* izh, State& state, double t, double dt) {
    izh_forward(static_cast<void*>(izh), state, t, dt);
}
inline void izh_reset(IzhikevichNeuron* izh) {
    izh_reset(static_cast<void*>(izh));
}
inline void izh_to_device(IzhikevichNeuron* izh, torch::Device device) {
    izh_to_device(static_cast<void*>(izh), device);
}

// Ops table
extern nrn_module_ops izh_ops;

// Wrap as generic module handle
inline NrnModule izh_as_module(IzhikevichNeuron* izh) {
    return NrnModule{static_cast<void*>(izh), &izh_ops};
}

} // namespace neuron
} // namespace nrn
