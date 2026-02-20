#pragma once

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {

/// Double-exponential (rise + decay) synapse.
///
/// Postsynaptic current has distinct rise and decay phases:
///
///     I = I_decay - I_rise
///     dI_rise / dt  = -I_rise  / tau_rise
///     dI_decay / dt = -I_decay / tau_decay
///     I_rise  += W * spike   (on presynaptic spike)
///     I_decay += W * spike   (on presynaptic spike)
///
class DoubleExponentialSynapseImpl
    : public Module<DoubleExponentialSynapseImpl> {
public:
    /// Construct from population size and options.
    explicit DoubleExponentialSynapseImpl(
        int64_t n,
        const DoubleExponentialSynapseOptions& opts = {});

    /// Re-initialize parameters and buffers.
    void reset() override;

    /// Advance one timestep: decay both components, deliver spikes.
    void forward(State& state, Time t, Duration dt) override;

    /// State variable names managed by this module.
    std::vector<std::string> state_vars() const override;

private:
    DoubleExponentialSynapseOptions opts_;
};

TORCH_MODULE(DoubleExponentialSynapse);

} // namespace nrn
