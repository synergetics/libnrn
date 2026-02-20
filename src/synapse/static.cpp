#include <nrn/synapse/static.h>

namespace nrn {

StaticSynapseImpl::StaticSynapseImpl(int64_t n,
                                     const StaticSynapseOptions& opts)
    : opts_(opts) {
    n_ = n;
}

void StaticSynapseImpl::reset() {
    // Static synapse has no internal state to reset.
}

void StaticSynapseImpl::forward(State& /*state*/, Time /*t*/, Duration /*dt*/) {
    TORCH_CHECK(false,
                "StaticSynapse::forward() not yet implemented — "
                "requires connectivity tensor and spike buffer integration");
}

std::vector<std::string> StaticSynapseImpl::state_vars() const {
    return {"I_syn"};
}

} // namespace nrn
