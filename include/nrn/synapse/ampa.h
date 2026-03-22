/// @file ampa.h
/// @brief AMPA receptor synapse with reversal potential.
///
/// @details
/// Models fast excitatory glutamatergic transmission using conductance-based
/// double-exponential kinetics:
/// @code
///   ds_rise  / dt = -s_rise  / tau_rise
///   ds_decay / dt = -s_decay / tau_decay
///   g = weight * (s_decay - s_rise)
///   I_syn = g * (v_post - e_rev)
///   on spike:  s_rise += 1;  s_decay += 1
/// @endcode
/// State variables published to @c State: @c "I_syn", @c "g",
/// @c "s_rise", @c "s_decay".
///
/// @see AMPAOptions, ampa_create(), ampa_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

/// @brief Internal state for an AMPA receptor synapse population.
struct AMPASynapse {
    int64_t n;         ///< Number of postsynaptic neurons.
    AMPAOptions options; ///< Synapse parameters.
};

/// @brief Allocate and initialise an AMPA synapse population.
/// @param n     Number of postsynaptic neurons.
/// @param opts  Synapse parameters.
/// @return Heap-allocated AMPASynapse; caller must call @c ampa_destroy().
AMPASynapse* ampa_create(int64_t n, AMPAOptions opts = {});

/// @brief Free an AMPASynapse.
void ampa_destroy(AMPASynapse* syn);

/// @name ops-table implementations
/// @{
void ampa_forward(void* self, State& state, double t, double dt); ///< @see nrn_ops::forward
void ampa_reset(void* self);                                       ///< @see nrn_ops::reset
const char** ampa_state_vars(void* self, int* count);              ///< @see nrn_ops::state_vars
int64_t ampa_size(void* self);                                     ///< @see nrn_ops::size
void ampa_to_device(void* self, torch::Device device);             ///< @see nrn_ops::to_device
/// @}

inline void ampa_forward(AMPASynapse* syn, State& state, double t, double dt) {
    ampa_forward(static_cast<void*>(syn), state, t, dt);
}
inline void ampa_reset(AMPASynapse* syn) {
    ampa_reset(static_cast<void*>(syn));
}
inline void ampa_to_device(AMPASynapse* syn, torch::Device device) {
    ampa_to_device(static_cast<void*>(syn), device);
}

extern nrn_ops ampa_ops; ///< Ops table for AMPASynapse.

/// @brief Wrap an AMPASynapse into a type-erased NrnModule handle.
inline NrnModule ampa_as_module(AMPASynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &ampa_ops};
}

} // namespace synapse
} // namespace nrn
