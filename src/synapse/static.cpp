#include <nrn/synapse/static.h>

namespace nrn {
namespace synapse {

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

nrn_ops static_synapse_ops = {
    .forward    = static_synapse_forward,
    .reset      = static_synapse_reset,
    .state_vars = static_synapse_state_vars,
    .size       = static_synapse_size,
    .to_device  = static_synapse_to_device,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

StaticSynapse* static_synapse_create(int64_t n, StaticSynapseOptions opts) {
    auto* syn = new StaticSynapse();
    syn->n = n;
    syn->options = std::move(opts);
    return syn;
}

void static_synapse_destroy(StaticSynapse* syn) {
    delete syn;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void static_synapse_forward(void* /*self*/, State& /*state*/, double /*t*/,
                            double /*dt*/) {
    TORCH_CHECK(false,
                "StaticSynapse::forward() not yet implemented -- "
                "requires connectivity tensor and spike buffer integration");
}

void static_synapse_reset(void* /*self*/) {
    // Static synapse has no internal state to reset.
}

static const char* static_synapse_var_names[] = {"I_syn"};

const char** static_synapse_state_vars(void* /*self*/, int* count) {
    *count = 1;
    return static_synapse_var_names;
}

int64_t static_synapse_size(void* self) {
    return static_cast<StaticSynapse*>(self)->n;
}

void static_synapse_to_device(void* /*self*/, torch::Device /*device*/) {
    // No tensors stored in the struct yet -- nothing to move.
}

} // namespace synapse
} // namespace nrn
