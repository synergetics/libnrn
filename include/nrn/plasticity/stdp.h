#pragma once

#include <nrn/plasticity/plasticity_rule.h>
#include <nrn/plasticity/options.h>

namespace nrn {

// Forward declaration to avoid circular include.
struct ConnectivityTensor;

/// Internal state for pair-based Spike-Timing-Dependent Plasticity (STDP).
///
/// Maintains per-synapse pre and post eligibility traces that decay
/// exponentially, and updates weights on each spike according to the
/// classic asymmetric STDP window.
///
/// Implements:
///
///     dw = A_plus  * pre_trace  * post_spike
///        - A_minus * post_trace * pre_spike
///
///     w += learning_rate * dw * structural_mask * modulatory_mask
///     w  = clamp(w, w_min, w_max)
///
/// Pre- and post-synaptic traces decay exponentially:
///     d(trace_pre)/dt  = -trace_pre  / tau_plus
///     d(trace_post)/dt = -trace_post / tau_minus
///
/// On spike: trace += 1.
///
/// See STDPOptions for the full parameter set.
struct STDPState {
    STDPOptions opts;

    // Cached per-timestep decay factors (computed from tau and dt on
    // first call to update(), or when dt changes).
    double cached_dt  = 0.0;
    double decay_pre  = 0.0;
    double decay_post = 0.0;
};

// ---------------------------------------------------------------------------
// Free functions operating on STDPState
// ---------------------------------------------------------------------------

/// Allocate and return a new STDPState on the heap.
STDPState* stdp_create(const STDPOptions& opts = {});

/// Free an STDPState allocated with stdp_create.
void stdp_destroy(STDPState* s);

/// Allocate trace_pre and trace_post tensors inside the
/// ConnectivityTensor, matching the shape of the weight tensor.
void stdp_initialize(void* self, ConnectivityTensor& conn);

/// Apply one STDP update step.
void stdp_update(void* self, ConnectivityTensor& conn,
                 const State& pre_state, const State& post_state,
                 double t, double dt);

/// Zero all eligibility traces and reset cached state.
void stdp_reset(void* self);

/// Access options (read-only).
const STDPOptions& stdp_options(const STDPState* s);

/// Ops table for STDP.
extern plasticity_ops stdp_ops;

/// Wrap an STDPState pointer into a type-erased PlasticityRule handle.
inline PlasticityRule stdp_as_rule(STDPState* s) {
    return PlasticityRule{static_cast<void*>(s), &stdp_ops};
}

} // namespace nrn
