/// @file izhikevich.h
/// @brief Izhikevich neuron model (dimensionless convention).
///
/// @details
/// Implements the two-variable spiking model from Izhikevich (2003):
/// @code
///   dv/dt = 0.04*v^2 + 5*v + 140 - u + I_syn
///   du/dt = a * (b*v - u)
///   if v >= v_peak:  spike = 1;  v = c;  u += d
/// @endcode
/// The model is dimensionless; @c dt is in seconds (SI) but converted to
/// milliseconds internally to match the published constants.
///
/// Despite its simplicity the model can reproduce a wide range of cortical
/// firing patterns (RS, IB, CH, FS, LTS, etc.) by varying @c a, @c b,
/// @c c, @c d.
///
/// @see IzhikevichOptions, izh_create(), izh_as_module()

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

/// @brief Internal state for a population of Izhikevich neurons.
///
/// @details
/// All tensors are shape @c [N] (SoA layout).
/// State variables published to @c State: @c "v", @c "u", @c "spike", @c "I_syn".
struct IzhikevichNeuron {
    int64_t n;                  ///< Number of neurons.
    IzhikevichOptions options;  ///< Scalar defaults used to fill parameter tensors.

    // State tensors [N]
    torch::Tensor v;      ///< Membrane potential (dimensionless).
    torch::Tensor u;      ///< Recovery variable (dimensionless).
    torch::Tensor spike;  ///< Binary spike indicator (0.0 or 1.0).
    torch::Tensor I_syn;  ///< Total synaptic input current (dimensionless).

    // Parameter tensors [N]
    torch::Tensor a;      ///< Recovery time scale.
    torch::Tensor b;      ///< Sensitivity of @c u to @c v.
    torch::Tensor c;      ///< Post-spike reset value of @c v.
    torch::Tensor d;      ///< Post-spike increment of @c u.
    torch::Tensor v_peak; ///< Spike cutoff value.
};

/// @brief Allocate and initialise an Izhikevich neuron population.
///
/// @param n     Number of neurons.
/// @param opts  Parameter options.
/// @return Heap-allocated IzhikevichNeuron; caller must call @c izh_destroy().
IzhikevichNeuron* izh_create(int64_t n, IzhikevichOptions opts = {});

/// @brief Free an IzhikevichNeuron allocated by @c izh_create().
/// @param izh  Pointer to the neuron population; may be null (no-op).
void izh_destroy(IzhikevichNeuron* izh);

/// @name ops-table implementations (void* overloads for NrnModule dispatch)
/// @{
void izh_forward(void* self, State& state, double t, double dt);  ///< @see nrn_ops::forward
void izh_reset(void* self);                                        ///< @see nrn_ops::reset
const char** izh_state_vars(void* self, int* count);               ///< @see nrn_ops::state_vars
int64_t izh_size(void* self);                                      ///< @see nrn_ops::size
void izh_to_device(void* self, torch::Device device);              ///< @see nrn_ops::to_device
/// @}

/// @brief Typed forward pass (convenience wrapper).
inline void izh_forward(IzhikevichNeuron* izh, State& state, double t, double dt) {
    izh_forward(static_cast<void*>(izh), state, t, dt);
}

/// @brief Typed reset (convenience wrapper).
inline void izh_reset(IzhikevichNeuron* izh) {
    izh_reset(static_cast<void*>(izh));
}

/// @brief Typed device transfer (convenience wrapper).
inline void izh_to_device(IzhikevichNeuron* izh, torch::Device device) {
    izh_to_device(static_cast<void*>(izh), device);
}

extern nrn_ops izh_ops; ///< Ops table for IzhikevichNeuron.

/// @brief Wrap an IzhikevichNeuron into a type-erased NrnModule handle.
/// @param izh  Pointer to an existing Izhikevich population.
/// @return NrnModule with @c izh_ops dispatch table.
inline NrnModule izh_as_module(IzhikevichNeuron* izh) {
    return NrnModule{static_cast<void*>(izh), &izh_ops};
}

} // namespace neuron
} // namespace nrn
