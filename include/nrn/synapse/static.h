#pragma once

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/synapse/options.h>

namespace nrn {

/// Static (fixed-weight) synapse.
///
/// The simplest synapse model: on a presynaptic spike the postsynaptic
/// current is incremented by the connection weight.  No temporal dynamics.
///
///     I_syn += W * spike
///
class StaticSynapseImpl : public Module<StaticSynapseImpl> {
public:
    /// Construct from population size and options.
    explicit StaticSynapseImpl(int64_t n,
                               const StaticSynapseOptions& opts = {});

    /// Re-initialize parameters and buffers.
    void reset() override;

    /// Advance one timestep: deliver synaptic current.
    void forward(State& state, Time t, Duration dt) override;

    /// State variable names managed by this module.
    std::vector<std::string> state_vars() const override;

private:
    StaticSynapseOptions opts_;
};

TORCH_MODULE(StaticSynapse);

} // namespace nrn
