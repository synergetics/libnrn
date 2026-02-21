#include <nrn/plasticity/stdp.h>
#include <nrn/connectivity/connectivity_tensor.h>

#include <cmath>

namespace nrn {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

STDPState* stdp_create(const STDPOptions& opts) {
    auto* s = new STDPState{};
    s->opts = opts;
    s->cached_dt  = 0.0;
    s->decay_pre  = 0.0;
    s->decay_post = 0.0;
    return s;
}

void stdp_destroy(STDPState* s) {
    delete s;
}

// ---------------------------------------------------------------------------
// Ops implementations (void* self -> STDPState*)
// ---------------------------------------------------------------------------

void stdp_initialize(void* self, ConnectivityTensor& conn) {
    // self is unused for initialization state, but we validate weights.
    (void)self;

    TORCH_CHECK(conn.weights.defined(),
                "stdp_initialize: weights tensor must be defined");

    // Allocate trace tensors with the same shape and device as weights,
    // initialized to zero.
    conn.trace_pre  = torch::zeros_like(conn.weights);
    conn.trace_post = torch::zeros_like(conn.weights);
}

void stdp_update(void* self, ConnectivityTensor& /*conn*/,
                 const State& /*pre_state*/, const State& /*post_state*/,
                 double /*t*/, double dt) {
    auto* s = static_cast<STDPState*>(self);

    // Cache decay factors when dt changes.
    if (dt != s->cached_dt) {
        s->cached_dt  = dt;
        s->decay_pre  = std::exp(-dt / s->opts.tau_plus());
        s->decay_post = std::exp(-dt / s->opts.tau_minus());
    }

    TORCH_CHECK(false,
                "stdp_update() not yet implemented -- "
                "requires spike tensor extraction from pre/post State "
                "and element-wise trace decay + weight update kernels");
}

void stdp_reset(void* self) {
    auto* s = static_cast<STDPState*>(self);

    // Traces live inside the ConnectivityTensor, not here.
    // A full reset requires access to the ConnectivityTensor:
    //   conn.trace_pre->zero_();
    //   conn.trace_post->zero_();
    // This function resets only local cached state.
    s->cached_dt  = 0.0;
    s->decay_pre  = 0.0;
    s->decay_post = 0.0;
}

// ---------------------------------------------------------------------------
// Read-only accessor
// ---------------------------------------------------------------------------

const STDPOptions& stdp_options(const STDPState* s) {
    return s->opts;
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

plasticity_ops stdp_ops = {
    stdp_initialize,
    stdp_update,
    stdp_reset,
};

} // namespace nrn
