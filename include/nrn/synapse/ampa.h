#pragma once

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {

/// AMPA receptor synapse with reversal potential.
///
/// Models fast excitatory glutamatergic transmission with
/// conductance-based dynamics:
///
///     ds_rise / dt  = -s_rise  / tau_rise
///     ds_decay / dt = -s_decay / tau_decay
///     g = W * (s_decay - s_rise)
///     I_syn = g * (V - E_rev)
///
/// On presynaptic spike:
///     s_rise  += 1
///     s_decay += 1
///
class AMPAImpl : public Module<AMPAImpl> {
public:
    /// Construct from population size and options.
    explicit AMPAImpl(int64_t n, const AMPAOptions& opts = {});

    /// Re-initialize parameters and buffers.
    void reset() override;

    /// Advance one timestep: decay gating variables, deliver spikes,
    /// compute conductance-based current.
    void forward(State& state, Time t, Duration dt) override;

    /// State variable names managed by this module.
    std::vector<std::string> state_vars() const override;

private:
    AMPAOptions opts_;
};

TORCH_MODULE(AMPA);

} // namespace nrn
