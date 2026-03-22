/// @file lif.h
/// @brief Leaky Integrate-and-Fire (LIF) neuron model.
///
/// @details
/// The LIF model integrates synaptic input current with a leaky RC membrane
/// and fires a spike whenever @c v crosses @c v_thresh, then resets to
/// @c v_reset and enters a hard refractory period of @c tau_ref.
///
/// Dynamics (forward Euler):
/// @code
///   if refractory > 0:
///       refractory -= dt
///   else:
///       dv/dt = -(v - v_rest) / tau_m  +  (I_syn + i_bg) / c_m
///   if v >= v_thresh:
///       spike = 1;  v = v_reset;  refractory = tau_ref
/// @endcode
///
/// All parameter tensors are shape @c [N] to support per-neuron heterogeneity.
///
/// @see LIFOptions, lif_create(), lif_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

/// @brief Internal state for a population of LIF neurons.
///
/// @details
/// All tensors are shape @c [N] (SoA layout).  The @c state field of the
/// owning Population mirrors these tensors by name.
///
/// State variables published to @c State: @c "v", @c "spike",
/// @c "refractory", @c "I_syn".
struct LIFNeuron {
    int64_t n;           ///< Number of neurons.
    LIFOptions options;  ///< Scalar defaults used to fill parameter tensors.

    // State tensors [N]
    torch::Tensor v;          ///< Membrane potential (V).
    torch::Tensor spike;      ///< Binary spike indicator (0.0 or 1.0).
    torch::Tensor refractory; ///< Remaining refractory time (s); 0 when not refractory.
    torch::Tensor I_syn;      ///< Total synaptic input current (A).

    // Parameter tensors [N]
    torch::Tensor v_rest;     ///< Resting membrane potential (V).
    torch::Tensor v_thresh;   ///< Spike threshold (V).
    torch::Tensor v_reset;    ///< Post-spike reset potential (V).
    torch::Tensor tau_m;      ///< Membrane time constant (s).
    torch::Tensor tau_ref;    ///< Absolute refractory period (s).
    torch::Tensor c_m;        ///< Membrane capacitance (F).
    torch::Tensor i_bg;       ///< Constant background current (A).
};

/// @brief Allocate and initialise a LIF neuron population.
///
/// @param n     Number of neurons.
/// @param opts  Parameter options (homogeneous defaults; supports per-neuron override after creation).
/// @return Heap-allocated LIFNeuron; caller must call @c lif_destroy().
LIFNeuron* lif_create(int64_t n, LIFOptions opts = {});

/// @brief Free a LIFNeuron allocated by @c lif_create().
/// @param lif  Pointer to the neuron population; may be null (no-op).
void lif_destroy(LIFNeuron* lif);

/// @name ops-table implementations (void* overloads for NrnModule dispatch)
/// @{
void lif_forward(void* self, State& state, double t, double dt);   ///< @see nrn_ops::forward
void lif_reset(void* self);                                         ///< @see nrn_ops::reset
const char** lif_state_vars(void* self, int* count);                ///< @see nrn_ops::state_vars
int64_t lif_size(void* self);                                       ///< @see nrn_ops::size
void lif_to_device(void* self, torch::Device device);               ///< @see nrn_ops::to_device
/// @}

/// @brief Typed forward pass (convenience wrapper).
/// @param lif    LIF neuron population.
/// @param state  Named tensor bag; reads @c I_syn, writes @c v / @c spike / @c refractory.
/// @param t      Current simulation time (s).
/// @param dt     Timestep duration (s).
inline void lif_forward(LIFNeuron* lif, State& state, double t, double dt) {
    lif_forward(static_cast<void*>(lif), state, t, dt);
}

/// @brief Typed reset (convenience wrapper).
inline void lif_reset(LIFNeuron* lif) {
    lif_reset(static_cast<void*>(lif));
}

/// @brief Typed device transfer (convenience wrapper).
inline void lif_to_device(LIFNeuron* lif, torch::Device device) {
    lif_to_device(static_cast<void*>(lif), device);
}

extern nrn_ops lif_ops; ///< Ops table for LIFNeuron.

/// @brief Wrap a LIFNeuron into a type-erased NrnModule handle.
/// @param lif  Pointer to an existing LIF population.
/// @return NrnModule with @c lif_ops dispatch table.
inline NrnModule lif_as_module(LIFNeuron* lif) {
    return NrnModule{static_cast<void*>(lif), &lif_ops};
}

} // namespace neuron
} // namespace nrn
