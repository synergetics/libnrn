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
// LIF — Leaky Integrate-and-Fire neuron model
// ============================================================================
//
// State variables (all shape [N]):
//   v          — membrane potential (V)
//   spike      — binary spike indicator (0 or 1)
//   refractory — remaining refractory time (s), 0 when not refractory
//   I_syn      — total synaptic input current (A)
//
// Dynamics (forward Euler, to be replaced by CUDA kernel):
//   if refractory > 0:
//       refractory -= dt
//   else:
//       dv/dt = (-(v - v_rest) + I_syn * tau_m / c_m + i_bg * tau_m / c_m) / tau_m
//   if v >= v_thresh:
//       spike = 1, v = v_reset, refractory = tau_ref
//
// Parameters are stored as 1-D tensors of shape [N] to allow per-neuron
// heterogeneity (even though they are initialized uniformly from options).
// ============================================================================
class LIFImpl : public nrn::Module<LIFImpl> {
public:
    /// Construct N LIF neurons with default options.
    explicit LIFImpl(int64_t n);

    /// Construct N LIF neurons with the given options.
    LIFImpl(int64_t n, LIFOptions options);

    /// Initialize / reinitialize all state and parameter tensors.
    void reset() override;

    /// Advance the neuron state by one timestep dt.
    void forward(nrn::State& state, nrn::Time t, nrn::Duration dt) override;

    /// Return the names of state variables managed by this module.
    std::vector<std::string> state_vars() const override;

    // ------------------------------------------------------------------
    // State tensors (registered as buffers — not trainable parameters)
    // ------------------------------------------------------------------
    torch::Tensor v;          ///< Membrane potential [N]
    torch::Tensor spike;      ///< Spike indicator    [N]
    torch::Tensor refractory; ///< Refractory timer   [N]
    torch::Tensor I_syn;      ///< Synaptic current   [N]

    // ------------------------------------------------------------------
    // Model parameter tensors (registered as buffers)
    // ------------------------------------------------------------------
    torch::Tensor v_rest;     ///< Resting potential   [N]
    torch::Tensor v_thresh;   ///< Spike threshold     [N]
    torch::Tensor v_reset;    ///< Reset potential      [N]
    torch::Tensor tau_m;      ///< Membrane time const  [N]
    torch::Tensor tau_ref;    ///< Refractory period    [N]
    torch::Tensor c_m;        ///< Membrane capacitance [N]
    torch::Tensor i_bg;       ///< Background current   [N]

private:
    LIFOptions options_;
};

TORCH_MODULE(LIF);

} // namespace neuron
} // namespace nrn
