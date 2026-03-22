/// @file adex.h
/// @brief Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
///
/// @details
/// Implements the AdEx model from Brette & Gerstner (2005):
/// @code
///   c_m * dv/dt = -g_l*(v - v_rest)
///               + g_l*delta_t*exp((v - v_thresh)/delta_t)
///               - w + I_syn + i_bg
///   tau_w * dw/dt = a*(v - v_rest) - w
///   if v >= v_peak:
///       spike = 1;  v = v_reset;  w += b;  refractory = tau_ref
/// @endcode
/// The exponential term produces a sharp action-potential upswing; the
/// adaptation variable @c w captures spike-frequency adaptation.
///
/// All parameter tensors are shape @c [N] for per-neuron heterogeneity.
///
/// @see AdExOptions, adex_create(), adex_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

/// @brief Internal state for a population of AdEx neurons.
///
/// @details
/// All tensors are shape @c [N] (SoA layout).
/// State variables published to @c State: @c "v", @c "w", @c "spike",
/// @c "refractory", @c "I_syn".
struct AdExNeuron {
    int64_t n;          ///< Number of neurons.
    AdExOptions options; ///< Scalar defaults used to fill parameter tensors.

    // State tensors [N]
    torch::Tensor v;          ///< Membrane potential (V).
    torch::Tensor w;          ///< Adaptation current (A).
    torch::Tensor spike;      ///< Binary spike indicator (0.0 or 1.0).
    torch::Tensor refractory; ///< Remaining refractory time (s).
    torch::Tensor I_syn;      ///< Total synaptic input current (A).

    // Parameter tensors [N]
    torch::Tensor v_rest;    ///< Resting membrane potential (V).
    torch::Tensor v_thresh;  ///< Effective spike threshold (V).
    torch::Tensor v_reset;   ///< Post-spike reset potential (V).
    torch::Tensor v_peak;    ///< Spike cutoff voltage (V).
    torch::Tensor tau_m;     ///< Membrane time constant (s).
    torch::Tensor tau_w;     ///< Adaptation time constant (s).
    torch::Tensor tau_ref;   ///< Absolute refractory period (s).
    torch::Tensor c_m;       ///< Membrane capacitance (F).
    torch::Tensor g_l;       ///< Leak conductance (S).
    torch::Tensor a;         ///< Subthreshold adaptation coupling (S).
    torch::Tensor b;         ///< Spike-triggered adaptation increment (A).
    torch::Tensor delta_t;   ///< Exponential slope factor (V).
    torch::Tensor i_bg;      ///< Constant background current (A).
};

/// @brief Allocate and initialise an AdEx neuron population.
///
/// @param n     Number of neurons.
/// @param opts  Parameter options.
/// @return Heap-allocated AdExNeuron; caller must call @c adex_destroy().
AdExNeuron* adex_create(int64_t n, AdExOptions opts = {});

/// @brief Free an AdExNeuron allocated by @c adex_create().
/// @param adex  Pointer to the neuron population; may be null (no-op).
void adex_destroy(AdExNeuron* adex);

/// @name ops-table implementations (void* overloads for NrnModule dispatch)
/// @{
void adex_forward(void* self, State& state, double t, double dt);  ///< @see nrn_ops::forward
void adex_reset(void* self);                                        ///< @see nrn_ops::reset
const char** adex_state_vars(void* self, int* count);               ///< @see nrn_ops::state_vars
int64_t adex_size(void* self);                                      ///< @see nrn_ops::size
void adex_to_device(void* self, torch::Device device);              ///< @see nrn_ops::to_device
/// @}

/// @brief Typed forward pass (convenience wrapper).
inline void adex_forward(AdExNeuron* adex, State& state, double t, double dt) {
    adex_forward(static_cast<void*>(adex), state, t, dt);
}

/// @brief Typed reset (convenience wrapper).
inline void adex_reset(AdExNeuron* adex) {
    adex_reset(static_cast<void*>(adex));
}

/// @brief Typed device transfer (convenience wrapper).
inline void adex_to_device(AdExNeuron* adex, torch::Device device) {
    adex_to_device(static_cast<void*>(adex), device);
}

extern nrn_ops adex_ops; ///< Ops table for AdExNeuron.

/// @brief Wrap an AdExNeuron into a type-erased NrnModule handle.
/// @param adex  Pointer to an existing AdEx population.
/// @return NrnModule with @c adex_ops dispatch table.
inline NrnModule adex_as_module(AdExNeuron* adex) {
    return NrnModule{static_cast<void*>(adex), &adex_ops};
}

} // namespace neuron
} // namespace nrn
