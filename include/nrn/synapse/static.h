/// @file static.h
/// @brief Static (fixed-weight, instantaneous) synapse model.
///
/// @details
/// The simplest synapse model.  On each presynaptic spike, postsynaptic
/// current is incremented by the connection weight with no temporal
/// filtering:
/// @code
///   I_syn += weight * spike
/// @endcode
/// State variables published to @c State: @c "I_syn".
///
/// @see StaticSynapseOptions, static_synapse_create(), static_synapse_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {
namespace synapse {

/// @brief Internal state for a static synapse.
struct StaticSynapse {
    int64_t n;                    ///< Number of neurons (postsynaptic).
    StaticSynapseOptions options; ///< Synapse parameters.
};

/// @brief Allocate and initialise a StaticSynapse.
/// @param n     Number of postsynaptic neurons.
/// @param opts  Synapse parameters.
/// @return Heap-allocated StaticSynapse; caller must call @c static_synapse_destroy().
StaticSynapse* static_synapse_create(int64_t n, StaticSynapseOptions opts = {});

/// @brief Free a StaticSynapse.
void static_synapse_destroy(StaticSynapse* syn);

/// @name ops-table implementations
/// @{
void static_synapse_forward(void* self, State& state, double t, double dt);  ///< @see nrn_ops::forward
void static_synapse_reset(void* self);                                        ///< @see nrn_ops::reset
const char** static_synapse_state_vars(void* self, int* count);               ///< @see nrn_ops::state_vars
int64_t static_synapse_size(void* self);                                      ///< @see nrn_ops::size
void static_synapse_to_device(void* self, torch::Device device);              ///< @see nrn_ops::to_device
/// @}

inline void static_synapse_forward(StaticSynapse* syn, State& state, double t, double dt) {
    static_synapse_forward(static_cast<void*>(syn), state, t, dt);
}
inline void static_synapse_reset(StaticSynapse* syn) {
    static_synapse_reset(static_cast<void*>(syn));
}
inline void static_synapse_to_device(StaticSynapse* syn, torch::Device device) {
    static_synapse_to_device(static_cast<void*>(syn), device);
}

extern nrn_ops static_synapse_ops; ///< Ops table for StaticSynapse.

/// @brief Wrap a StaticSynapse into a type-erased NrnModule handle.
inline NrnModule static_synapse_as_module(StaticSynapse* syn) {
    return NrnModule{static_cast<void*>(syn), &static_synapse_ops};
}

} // namespace synapse
} // namespace nrn
