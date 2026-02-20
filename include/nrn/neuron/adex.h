#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/neuron/options.h>

namespace nrn {
namespace neuron {

// ============================================================================
// AdEx — Adaptive Exponential Integrate-and-Fire neuron model
// ============================================================================
//
// State variables (all shape [N]):
//   v          — membrane potential (V)
//   w          — adaptation current (A)
//   spike      — binary spike indicator (0 or 1)
//   refractory — remaining refractory time (s)
//   I_syn      — total synaptic input current (A)
//
// Dynamics (Brette & Gerstner 2005):
//   c_m * dv/dt = -g_l*(v - v_rest) + g_l*delta_t*exp((v - v_thresh)/delta_t)
//                 - w + I_syn + i_bg
//   tau_w * dw/dt = a*(v - v_rest) - w
//
//   if v >= v_peak:
//       spike = 1
//       v = v_reset
//       w += b
//       refractory = tau_ref
//
// Parameters stored as 1-D tensors [N] for per-neuron heterogeneity.
// ============================================================================
class AdExImpl : public nrn::Module<AdExImpl> {
public:
    /// Construct N AdEx neurons with default options.
    explicit AdExImpl(int64_t n);

    /// Construct N AdEx neurons with the given options.
    AdExImpl(int64_t n, AdExOptions options);

    /// Initialize / reinitialize all state and parameter tensors.
    void reset() override;

    /// Advance the neuron state by one timestep dt.
    void forward(nrn::State& state, nrn::Time t, nrn::Duration dt) override;

    /// Return the names of state variables managed by this module.
    std::vector<std::string> state_vars() const override;

    // ------------------------------------------------------------------
    // State tensors (registered as buffers)
    // ------------------------------------------------------------------
    torch::Tensor v;          ///< Membrane potential   [N]
    torch::Tensor w;          ///< Adaptation current   [N]
    torch::Tensor spike;      ///< Spike indicator      [N]
    torch::Tensor refractory; ///< Refractory timer     [N]
    torch::Tensor I_syn;      ///< Synaptic current     [N]

    // ------------------------------------------------------------------
    // Model parameter tensors (registered as buffers)
    // ------------------------------------------------------------------
    torch::Tensor v_rest;     ///< Resting potential        [N]
    torch::Tensor v_thresh;   ///< Effective threshold      [N]
    torch::Tensor v_reset;    ///< Reset potential           [N]
    torch::Tensor v_peak;     ///< Spike cutoff / peak      [N]
    torch::Tensor tau_m;      ///< Membrane time constant   [N]
    torch::Tensor tau_w;      ///< Adaptation time constant [N]
    torch::Tensor tau_ref;    ///< Refractory period        [N]
    torch::Tensor c_m;        ///< Membrane capacitance     [N]
    torch::Tensor g_l;        ///< Leak conductance         [N]
    torch::Tensor a;          ///< Subthreshold adaptation  [N]
    torch::Tensor b;          ///< Spike-triggered adapt.   [N]
    torch::Tensor delta_t;    ///< Exponential slope factor [N]
    torch::Tensor i_bg;       ///< Background current       [N]

private:
    AdExOptions options_;
};

TORCH_MODULE(AdEx);

} // namespace neuron
} // namespace nrn
