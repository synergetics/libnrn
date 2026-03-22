/// @file double_exponential.h
/// @brief Double-exponential (rise + decay) synapse model.
///
/// @details
/// Models a postsynaptic current with distinct rise and decay phases:
/// @code
///   I = I_decay - I_rise
///   dI_rise  / dt = -I_rise  / tau_rise
///   dI_decay / dt = -I_decay / tau_decay
///   on spike:  I_rise += weight;  I_decay += weight
/// @endcode
/// This produces a realistic synaptic waveform with a finite rise time.
/// State variables published to @c State: @c "I_syn", @c "I_rise", @c "I_decay".
///
/// @see DoubleExponentialSynapseOptions, dexp_synapse_create(), dexp_synapse_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

/// @brief Internal state for a double-exponential synapse.
struct DoubleExponentialSynapse {
    int64_t n;                                ///< Number of neurons.
    DoubleExponentialSynapseOptions options;  ///< Synapse parameters.
};

/// @brief Allocate and initialise a DoubleExponentialSynapse.
/// @param n     Number of postsynaptic neurons.
/// @param opts  Synapse parameters.
/// @return Heap-allocated DoubleExponentialSynapse; caller must call @c dexp_synapse_destroy().
DoubleExponentialSynapse* dexp_synapse_create(int64_t n, DoubleExponentialSynapseOptions opts = {});

/// @brief Free a DoubleExponentialSynapse.
void dexp_synapse_destroy(DoubleExponentialSynapse* syn);

/// @name ops-table implementations
/// @{
void dexp_synapse_forward(void* self, State& state, double t, double dt); ///< @see nrn_ops::forward
void dexp_synapse_reset(void* self);                                       ///< @see nrn_ops::reset
const char** dexp_synapse_state_vars(void* self, int* count);              ///< @see nrn_ops::state_vars
int64_t dexp_synapse_size(void* self);                                     ///< @see nrn_ops::size
void dexp_synapse_to_device(void* self, torch::Device device);             ///< @see nrn_ops::to_device
/// @}

inline void dexp_synapse_forward(DoubleExponentialSynapse* syn, State& state, double t, double dt) {
    dexp_synapse_forward(static_cast<void*>(syn), state, t, dt);
}
inline void dexp_synapse_reset(DoubleExponentialSynapse* syn) {
    dexp_synapse_reset(static_cast<void*>(syn));
}
inline void dexp_synapse_to_device(DoubleExponentialSynapse* syn, torch::Device device) {
    dexp_synapse_to_device(static_cast<void*>(syn), device);
}

extern nrn_ops dexp_synapse_ops; ///< Ops table for DoubleExponentialSynapse.

/// @brief Wrap a DoubleExponentialSynapse into a type-erased NrnModule handle.
inline NrnModule dexp_synapse_as_module(DoubleExponentialSynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &dexp_synapse_ops};
}

} // namespace synapse
} // namespace nrn
