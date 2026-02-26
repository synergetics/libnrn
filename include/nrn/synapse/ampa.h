#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

// ============================================================================
// AMPASynapse -- AMPA receptor synapse with reversal potential
// ============================================================================
//
// Models fast excitatory glutamatergic transmission with
// conductance-based dynamics:
//
//     ds_rise / dt  = -s_rise  / tau_rise
//     ds_decay / dt = -s_decay / tau_decay
//     g = W * (s_decay - s_rise)
//     I_syn = g * (V - E_rev)
//
// On presynaptic spike:
//     s_rise  += 1
//     s_decay += 1
//
// State variable names: {"I_syn", "g", "s_rise", "s_decay"}
// ============================================================================

struct AMPASynapse {
    int64_t n;
    AMPAOptions options;
};

// Lifecycle
AMPASynapse* ampa_create(int64_t n, AMPAOptions opts = {});
void ampa_destroy(AMPASynapse* syn);

// Operations (void* self for ops table compatibility)
void ampa_forward(void* self, State& state, double t, double dt);
void ampa_reset(void* self);
const char** ampa_state_vars(void* self, int* count);
int64_t ampa_size(void* self);
void ampa_to_device(void* self, torch::Device device);

// Typed convenience wrappers
inline void ampa_forward(AMPASynapse* syn, State& state, double t, double dt) {
    ampa_forward(static_cast<void*>(syn), state, t, dt);
}
inline void ampa_reset(AMPASynapse* syn) {
    ampa_reset(static_cast<void*>(syn));
}
inline void ampa_to_device(AMPASynapse* syn, torch::Device device) {
    ampa_to_device(static_cast<void*>(syn), device);
}

// Ops table
extern nrn_ops ampa_ops;

// Wrap as generic module handle
inline NrnModule ampa_as_module(AMPASynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &ampa_ops};
}

} // namespace synapse
} // namespace nrn
