#include <nrn/synapse/exponential.h>

namespace nrn {
namespace synapse {

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_module_ops exp_synapse_ops = {
    .forward    = exp_synapse_forward,
    .reset      = exp_synapse_reset,
    .state_vars = exp_synapse_state_vars,
    .size       = exp_synapse_size,
    .to_device  = exp_synapse_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ExponentialSynapse* exp_synapse_create(int64_t n, ExponentialSynapseOptions opts) {
    auto* syn = new ExponentialSynapse();
    syn->n = n;
    syn->options = std::move(opts);
    return syn;
}

void exp_synapse_destroy(ExponentialSynapse* syn) {
    delete syn;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void exp_synapse_forward(void* /*self*/, State& /*state*/, double /*t*/,
                         double /*dt*/) {
    TORCH_CHECK(false,
                "ExponentialSynapse::forward() not yet implemented -- "
                "requires connectivity tensor and spike buffer integration");
}

void exp_synapse_reset(void* /*self*/) {
    // Will zero-initialize the current buffer when state management
    // is connected.
}

static const char* exp_synapse_var_names[] = {"I_syn"};

const char** exp_synapse_state_vars(void* /*self*/, int* count) {
    *count = 1;
    return exp_synapse_var_names;
}

int64_t exp_synapse_size(void* self) {
    return static_cast<ExponentialSynapse*>(self)->n;
}

void exp_synapse_to_device(void* /*self*/, torch::Device /*device*/) {
    // No tensors stored in the struct yet -- nothing to move.
}

} // namespace synapse
} // namespace nrn
