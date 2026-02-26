#include <nrn/synapse/double_exponential.h>

namespace nrn {
namespace synapse {

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_ops dexp_synapse_ops = {
    .forward    = dexp_synapse_forward,
    .reset      = dexp_synapse_reset,
    .state_vars = dexp_synapse_state_vars,
    .size       = dexp_synapse_size,
    .to_device  = dexp_synapse_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

DoubleExponentialSynapse* dexp_synapse_create(int64_t n,
                                              DoubleExponentialSynapseOptions opts) {
    auto* syn = new DoubleExponentialSynapse();
    syn->n = n;
    syn->options = std::move(opts);
    return syn;
}

void dexp_synapse_destroy(DoubleExponentialSynapse* syn) {
    delete syn;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void dexp_synapse_forward(void* /*self*/, State& /*state*/, double /*t*/,
                          double /*dt*/) {
    TORCH_CHECK(false,
                "DoubleExponentialSynapse::forward() not yet implemented -- "
                "requires connectivity tensor and spike buffer integration");
}

void dexp_synapse_reset(void* /*self*/) {
    // Will zero-initialize rise and decay current buffers when state
    // management is connected.
}

static const char* dexp_synapse_var_names[] = {"I_syn", "I_rise", "I_decay"};

const char** dexp_synapse_state_vars(void* /*self*/, int* count) {
    *count = 3;
    return dexp_synapse_var_names;
}

int64_t dexp_synapse_size(void* self) {
    return static_cast<DoubleExponentialSynapse*>(self)->n;
}

void dexp_synapse_to_device(void* /*self*/, torch::Device /*device*/) {
    // No tensors stored in the struct yet -- nothing to move.
}

} // namespace synapse
} // namespace nrn
