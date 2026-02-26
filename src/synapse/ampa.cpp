#include <nrn/synapse/ampa.h>

namespace nrn {
namespace synapse {

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_ops ampa_ops = {
    .forward    = ampa_forward,
    .reset      = ampa_reset,
    .state_vars = ampa_state_vars,
    .size       = ampa_size,
    .to_device  = ampa_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

AMPASynapse* ampa_create(int64_t n, AMPAOptions opts) {
    auto* syn = new AMPASynapse();
    syn->n = n;
    syn->options = std::move(opts);
    return syn;
}

void ampa_destroy(AMPASynapse* syn) {
    delete syn;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void ampa_forward(void* /*self*/, State& /*state*/, double /*t*/,
                  double /*dt*/) {
    TORCH_CHECK(false,
                "AMPA::forward() not yet implemented -- "
                "requires connectivity tensor, spike buffer, and "
                "postsynaptic voltage for conductance-based current");
}

void ampa_reset(void* /*self*/) {
    // Will zero-initialize gating variables and conductance buffers
    // when state management is connected.
}

static const char* ampa_var_names[] = {"I_syn", "g", "s_rise", "s_decay"};

const char** ampa_state_vars(void* /*self*/, int* count) {
    *count = 4;
    return ampa_var_names;
}

int64_t ampa_size(void* self) {
    return static_cast<AMPASynapse*>(self)->n;
}

void ampa_to_device(void* /*self*/, torch::Device /*device*/) {
    // No tensors stored in the struct yet -- nothing to move.
}

} // namespace synapse
} // namespace nrn
