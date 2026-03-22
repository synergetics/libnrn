/// @file exponential.h
/// @brief Single-exponential decay synapse model.
///
/// @details
/// Postsynaptic current decays with a single exponential time constant:
/// @code
///   dI/dt = -I / tau_decay
///   I    += weight * spike   (on presynaptic spike)
/// @endcode
/// State variables published to @c State: @c "I_syn".
///
/// @see ExponentialSynapseOptions, exp_synapse_create(), exp_synapse_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

/// @brief Internal state for a single-exponential decay synapse.
struct ExponentialSynapse {
    int64_t n;                         ///< Number of neurons.
    ExponentialSynapseOptions options; ///< Synapse parameters.
};

/// @brief Allocate and initialise an ExponentialSynapse.
/// @param n     Number of postsynaptic neurons.
/// @param opts  Synapse parameters.
/// @return Heap-allocated ExponentialSynapse; caller must call @c exp_synapse_destroy().
ExponentialSynapse* exp_synapse_create(int64_t n, ExponentialSynapseOptions opts = {});

/// @brief Free an ExponentialSynapse.
void exp_synapse_destroy(ExponentialSynapse* syn);

/// @name ops-table implementations
/// @{
void exp_synapse_forward(void* self, State& state, double t, double dt); ///< @see nrn_ops::forward
void exp_synapse_reset(void* self);                                       ///< @see nrn_ops::reset
const char** exp_synapse_state_vars(void* self, int* count);              ///< @see nrn_ops::state_vars
int64_t exp_synapse_size(void* self);                                     ///< @see nrn_ops::size
void exp_synapse_to_device(void* self, torch::Device device);             ///< @see nrn_ops::to_device
/// @}

inline void exp_synapse_forward(ExponentialSynapse* syn, State& state, double t, double dt) {
    exp_synapse_forward(static_cast<void*>(syn), state, t, dt);
}
inline void exp_synapse_reset(ExponentialSynapse* syn) {
    exp_synapse_reset(static_cast<void*>(syn));
}
inline void exp_synapse_to_device(ExponentialSynapse* syn, torch::Device device) {
    exp_synapse_to_device(static_cast<void*>(syn), device);
}

extern nrn_ops exp_synapse_ops; ///< Ops table for ExponentialSynapse.

/// @brief Wrap an ExponentialSynapse into a type-erased NrnModule handle.
inline NrnModule exp_synapse_as_module(ExponentialSynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &exp_synapse_ops};
}

} // namespace synapse
} // namespace nrn
