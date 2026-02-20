#include <nrn/synapse/exponential.h>

namespace nrn {

ExponentialSynapseImpl::ExponentialSynapseImpl(
    int64_t n,
    const ExponentialSynapseOptions& opts)
    : opts_(opts) {
    n_ = n;
}

void ExponentialSynapseImpl::reset() {
    // Will zero-initialize the current buffer when state management
    // is connected.
}

void ExponentialSynapseImpl::forward(State& /*state*/, Time /*t*/,
                                     Duration /*dt*/) {
    TORCH_CHECK(false,
                "ExponentialSynapse::forward() not yet implemented — "
                "requires connectivity tensor and spike buffer integration");
}

std::vector<std::string> ExponentialSynapseImpl::state_vars() const {
    return {"I_syn"};
}

} // namespace nrn
