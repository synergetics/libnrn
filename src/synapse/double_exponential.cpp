#include <nrn/synapse/double_exponential.h>

namespace nrn {

DoubleExponentialSynapseImpl::DoubleExponentialSynapseImpl(
    int64_t n,
    const DoubleExponentialSynapseOptions& opts)
    : opts_(opts) {
    n_ = n;
}

void DoubleExponentialSynapseImpl::reset() {
    // Will zero-initialize rise and decay current buffers when state
    // management is connected.
}

void DoubleExponentialSynapseImpl::forward(State& /*state*/, Time /*t*/,
                                           Duration /*dt*/) {
    TORCH_CHECK(false,
                "DoubleExponentialSynapse::forward() not yet implemented — "
                "requires connectivity tensor and spike buffer integration");
}

std::vector<std::string> DoubleExponentialSynapseImpl::state_vars() const {
    return {"I_syn", "I_rise", "I_decay"};
}

} // namespace nrn
