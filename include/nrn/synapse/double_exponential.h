#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

// ============================================================================
// DoubleExponentialSynapse -- Rise + decay synapse model
// ============================================================================
//
// Postsynaptic current has distinct rise and decay phases:
//
//     I = I_decay - I_rise
//     dI_rise / dt  = -I_rise  / tau_rise
//     dI_decay / dt = -I_decay / tau_decay
//     I_rise  += W * spike   (on presynaptic spike)
//     I_decay += W * spike   (on presynaptic spike)
//
// State variable names: {"I_syn", "I_rise", "I_decay"}
// ============================================================================

struct DoubleExponentialSynapse {
    int64_t n;
    DoubleExponentialSynapseOptions options;
};

// Lifecycle
DoubleExponentialSynapse* dexp_synapse_create(int64_t n, DoubleExponentialSynapseOptions opts = {});
void dexp_synapse_destroy(DoubleExponentialSynapse* syn);

// Operations (void* self for ops table compatibility)
void dexp_synapse_forward(void* self, State& state, double t, double dt);
void dexp_synapse_reset(void* self);
const char** dexp_synapse_state_vars(void* self, int* count);
int64_t dexp_synapse_size(void* self);
void dexp_synapse_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void dexp_synapse_forward(DoubleExponentialSynapse* syn, State& state, double t, double dt) {
    dexp_synapse_forward(static_cast<void*>(syn), state, t, dt);
}
inline void dexp_synapse_reset(DoubleExponentialSynapse* syn) {
    dexp_synapse_reset(static_cast<void*>(syn));
}
inline void dexp_synapse_to_device(DoubleExponentialSynapse* syn, torch::Device device) {
    dexp_synapse_to_device(static_cast<void*>(syn), device);
}

// Ops table
extern nrn_module_ops dexp_synapse_ops;

// Wrap as generic module handle
inline NrnModule dexp_synapse_as_module(DoubleExponentialSynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &dexp_synapse_ops};
}

} // namespace synapse
} // namespace nrn
