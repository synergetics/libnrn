#pragma once

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {

/// Exponential-decay synapse.
///
/// Postsynaptic current decays with a single exponential time constant:
///
///     dI/dt = -I / tau_decay
///     I += W * spike   (on presynaptic spike)
///
class ExponentialSynapseImpl : public Module<ExponentialSynapseImpl> {
public:
    /// Construct from population size and options.
    explicit ExponentialSynapseImpl(int64_t n,
                                   const ExponentialSynapseOptions& opts = {});

    /// Re-initialize parameters and buffers.
    void reset() override;

    /// Advance one timestep: decay current, deliver spikes.
    void forward(State& state, Time t, Duration dt) override;

    /// State variable names managed by this module.
    std::vector<std::string> state_vars() const override;

private:
    ExponentialSynapseOptions opts_;
};

TORCH_MODULE(ExponentialSynapse);

} // namespace nrn
