#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

// ============================================================================
// StaticSynapse -- Fixed-weight synapse model
// ============================================================================
//
// The simplest synapse model: on a presynaptic spike the postsynaptic
// current is incremented by the connection weight.  No temporal dynamics.
//
//     I_syn += W * spike
//
// State variable names: {"I_syn"}
// ============================================================================

struct StaticSynapse {
    int64_t n;
    StaticSynapseOptions options;
};

// Lifecycle
StaticSynapse* static_synapse_create(int64_t n, StaticSynapseOptions opts = {});
void static_synapse_destroy(StaticSynapse* syn);

// Operations (void* self for ops table compatibility)
void static_synapse_forward(void* self, State& state, double t, double dt);
void static_synapse_reset(void* self);
const char** static_synapse_state_vars(void* self, int* count);
int64_t static_synapse_size(void* self);
void static_synapse_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void static_synapse_forward(StaticSynapse* syn, State& state, double t, double dt) {
    static_synapse_forward(static_cast<void*>(syn), state, t, dt);
}
inline void static_synapse_reset(StaticSynapse* syn) {
    static_synapse_reset(static_cast<void*>(syn));
}
inline void static_synapse_to_device(StaticSynapse* syn, torch::Device device) {
    static_synapse_to_device(static_cast<void*>(syn), device);
}

// Ops table
extern nrn_ops static_synapse_ops;

// Wrap as generic module handle
inline NrnModule static_synapse_as_module(StaticSynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &static_synapse_ops};
}

} // namespace synapse
} // namespace nrn
