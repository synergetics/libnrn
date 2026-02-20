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
// Izhikevich neuron model (dimensionless convention)
// ============================================================================
//
// State variables (all shape [N]):
//   v     — membrane potential (dimensionless, mV-like)
//   u     — recovery variable  (dimensionless)
//   spike — binary spike indicator (0 or 1)
//   I_syn — total synaptic input current (dimensionless)
//
// Dynamics (Izhikevich 2003):
//   dv/dt = 0.04*v^2 + 5*v + 140 - u + I_syn
//   du/dt = a * (b*v - u)
//
//   if v >= v_peak:
//       spike = 1
//       v = c
//       u = u + d
//
// Note: dt in the forward() call is in seconds (SI), but the Izhikevich model
// uses millisecond-scale dynamics internally. The kernel handles the conversion.
//
// Parameters stored as 1-D tensors [N] for per-neuron heterogeneity.
// ============================================================================
class IzhikevichImpl : public nrn::Module<IzhikevichImpl> {
public:
    /// Construct N Izhikevich neurons with default options (Regular Spiking).
    explicit IzhikevichImpl(int64_t n);

    /// Construct N Izhikevich neurons with the given options.
    IzhikevichImpl(int64_t n, IzhikevichOptions options);

    /// Initialize / reinitialize all state and parameter tensors.
    void reset() override;

    /// Advance the neuron state by one timestep dt.
    void forward(nrn::State& state, nrn::Time t, nrn::Duration dt) override;

    /// Return the names of state variables managed by this module.
    std::vector<std::string> state_vars() const override;

    // ------------------------------------------------------------------
    // State tensors (registered as buffers)
    // ------------------------------------------------------------------
    torch::Tensor v;      ///< Membrane potential  [N]
    torch::Tensor u;      ///< Recovery variable   [N]
    torch::Tensor spike;  ///< Spike indicator     [N]
    torch::Tensor I_syn;  ///< Synaptic current    [N]

    // ------------------------------------------------------------------
    // Model parameter tensors (registered as buffers)
    // ------------------------------------------------------------------
    torch::Tensor a;      ///< Recovery time scale          [N]
    torch::Tensor b;      ///< Recovery sensitivity to v    [N]
    torch::Tensor c;      ///< Post-spike reset of v        [N]
    torch::Tensor d;      ///< Post-spike increment of u    [N]
    torch::Tensor v_peak; ///< Spike cutoff                 [N]

private:
    IzhikevichOptions options_;
};

TORCH_MODULE(Izhikevich);

} // namespace neuron
} // namespace nrn
