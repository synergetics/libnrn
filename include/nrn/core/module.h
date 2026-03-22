/// @file module.h
/// @brief Type-erased NrnModule handle and the nrn_ops dispatch table.
///
/// @details
/// libnrn avoids virtual dispatch and @c torch::nn::Module inheritance.
/// Instead, each concrete model type (LIFNeuron, AdExNeuron, …) provides a
/// static @c nrn_ops instance containing function pointers to its
/// implementations, and is wrapped in an @c NrnModule value consisting of
/// an opaque @c void* plus a pointer to that ops table.
///
/// This is the Linux-kernel-style ops table pattern: polymorphism without
/// the overhead of C++ virtual dispatch or RTTI.
///
/// @see nrn_forward(), nrn_reset(), NrnModule, nrn_ops

#pragma once

#include <cstdint>
#include <string>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>

namespace nrn {

/// @brief Ops table for any NrnModule concrete type.
///
/// @details
/// Every concrete module (neuron model, synapse model) provides a @b static
/// instance of this struct.  Function pointers accept @c void* @c self so
/// that the same table format works for all types without templates.
///
/// @see NrnModule
struct nrn_ops {
    /// @brief Advance the model by one timestep, updating @p state in place.
    ///
    /// @param self   Opaque pointer to the concrete model struct.
    /// @param state  Named tensor bag; read inputs, write outputs.
    /// @param t      Current simulation time in seconds.
    /// @param dt     Timestep duration in seconds.
    void (*forward)(void* self, State& state, double t, double dt);

    /// @brief Reset all internal tensors to initial conditions.
    ///
    /// @param self  Opaque pointer to the concrete model struct.
    void (*reset)(void* self);

    /// @brief Return the names of the state variables this module publishes.
    ///
    /// @param self   Opaque pointer to the concrete model struct.
    /// @param count  Output: number of names in the returned array.
    /// @return Pointer to a null-terminated array of C strings.
    const char** (*state_vars)(void* self, int* count);

    /// @brief Return the number of neurons (or synapses) this module owns.
    ///
    /// @param self  Opaque pointer to the concrete model struct.
    /// @return Number of elements.
    int64_t (*size)(void* self);

    /// @brief Move all internal tensors to @p device.
    ///
    /// @param self    Opaque pointer to the concrete model struct.
    /// @param device  Target torch device.
    void (*to_device)(void* self, torch::Device device);
};

/// @brief Type-erased handle to any libnrn module.
///
/// @details
/// Holds an opaque @c void* to the concrete struct (e.g. @c LIFNeuron*) and
/// a pointer to the corresponding ops table.  Population, NodeGroup, and
/// EdgeGroup store this handle and dispatch through it without knowing the
/// concrete type.
///
/// Wrap a concrete model with e.g. @c lif_as_module(), @c adex_as_module().
///
/// @see nrn_ops, lif_as_module(), adex_as_module(), izh_as_module()
struct NrnModule {
    void* impl;    ///< Concrete model pointer (LIFNeuron*, AdExNeuron*, …).
    nrn_ops* ops;  ///< Dispatch table for this module type.
};

// ---------------------------------------------------------------------------
// Dispatch convenience functions
// ---------------------------------------------------------------------------

/// @brief Call @c forward on the module.
/// @see nrn_ops::forward
inline void nrn_forward(NrnModule* m, State& state, double t, double dt) {
    m->ops->forward(m->impl, state, t, dt);
}

/// @brief Call @c reset on the module.
/// @see nrn_ops::reset
inline void nrn_reset(NrnModule* m) {
    m->ops->reset(m->impl);
}

/// @brief Return the number of neurons/synapses owned by the module.
/// @see nrn_ops::size
inline int64_t nrn_size(NrnModule* m) {
    return m->ops->size(m->impl);
}

/// @brief Move all module tensors to @p device.
/// @see nrn_ops::to_device
inline void nrn_to_device(NrnModule* m, torch::Device device) {
    m->ops->to_device(m->impl, device);
}

/// @brief Return the names of state variables published by this module.
///
/// @param m      Module handle.
/// @param count  Output: number of names in the returned array.
/// @return Array of C strings; valid for the lifetime of the module.
/// @see nrn_ops::state_vars
inline const char** nrn_state_vars(NrnModule* m, int* count) {
    return m->ops->state_vars(m->impl, count);
}

} // namespace nrn
