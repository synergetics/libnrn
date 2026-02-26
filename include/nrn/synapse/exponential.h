#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

// ============================================================================
// ExponentialSynapse -- Single-exponential decay synapse model
// ============================================================================
//
// Postsynaptic current decays with a single exponential time constant:
//
//     dI/dt = -I / tau_decay
//     I += W * spike   (on presynaptic spike)
//
// State variable names: {"I_syn"}
// ============================================================================

struct ExponentialSynapse {
    int64_t n;
    ExponentialSynapseOptions options;
};

// Lifecycle
ExponentialSynapse* exp_synapse_create(int64_t n, ExponentialSynapseOptions opts = {});
void exp_synapse_destroy(ExponentialSynapse* syn);

// Operations (void* self for ops table compatibility)
void exp_synapse_forward(void* self, State& state, double t, double dt);
void exp_synapse_reset(void* self);
const char** exp_synapse_state_vars(void* self, int* count);
int64_t exp_synapse_size(void* self);
void exp_synapse_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void exp_synapse_forward(ExponentialSynapse* syn, State& state, double t, double dt) {
    exp_synapse_forward(static_cast<void*>(syn), state, t, dt);
}
inline void exp_synapse_reset(ExponentialSynapse* syn) {
    exp_synapse_reset(static_cast<void*>(syn));
}
inline void exp_synapse_to_device(ExponentialSynapse* syn, torch::Device device) {
    exp_synapse_to_device(static_cast<void*>(syn), device);
}

// Ops table
extern nrn_ops exp_synapse_ops;

// Wrap as generic module handle
inline NrnModule exp_synapse_as_module(ExponentialSynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &exp_synapse_ops};
}

} // namespace synapse
} // namespace nrn
