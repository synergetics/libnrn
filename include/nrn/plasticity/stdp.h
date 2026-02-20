#pragma once

#include <nrn/plasticity/plasticity_rule.h>
#include <nrn/plasticity/options.h>

namespace nrn {

// Forward declaration to avoid circular include.
struct ConnectivityTensor;

/// Pair-based Spike-Timing-Dependent Plasticity (STDP).
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
class STDP : public PlasticityRule {
public:
    explicit STDP(const STDPOptions& opts = {});

    /// Allocate trace_pre and trace_post tensors inside the
    /// ConnectivityTensor, matching the shape of the weight tensor.
    void initialize(ConnectivityTensor& conn) override;

    /// Apply one STDP update step.
    void update(ConnectivityTensor& conn,
                const State& pre_state,
                const State& post_state,
                Time t,
                Duration dt) override;

    /// Zero all eligibility traces.
    void reset() override;

    /// Access options.
    const STDPOptions& options() const { return opts_; }

private:
    STDPOptions opts_;

    // Cached per-timestep decay factors (computed from tau and dt on
    // first call to update(), or when dt changes).
    double cached_dt_  = 0.0;
    double decay_pre_  = 0.0;
    double decay_post_ = 0.0;
};

} // namespace nrn
