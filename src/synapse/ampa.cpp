#include <nrn/synapse/ampa.h>

namespace nrn {

AMPAImpl::AMPAImpl(int64_t n, const AMPAOptions& opts)
    : opts_(opts) {
    n_ = n;
}

void AMPAImpl::reset() {
    // Will zero-initialize gating variables and conductance buffers
    // when state management is connected.
}

void AMPAImpl::forward(State& /*state*/, Time /*t*/, Duration /*dt*/) {
    TORCH_CHECK(false,
                "AMPA::forward() not yet implemented — "
                "requires connectivity tensor, spike buffer, and "
                "postsynaptic voltage for conductance-based current");
}

std::vector<std::string> AMPAImpl::state_vars() const {
    return {"I_syn", "g", "s_rise", "s_decay"};
}

} // namespace nrn
