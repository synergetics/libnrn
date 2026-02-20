#pragma once

#include <nrn/core/types.h>
#include <nrn/core/state.h>

namespace nrn {

// Forward declaration — avoid pulling in the full connectivity header.
struct ConnectivityTensor;

/// Abstract base for synaptic plasticity rules.
///
/// PlasticityRule is intentionally NOT a Module.  Plasticity rules
/// operate *on* connectivity tensors rather than maintaining their own
/// state tensors — they are attached to a Connection and invoked by
/// the simulation loop at the appropriate timescale.
///
/// Concrete rules (STDP, BCM, homeostatic scaling, etc.) implement
/// the three pure-virtual methods below.
class PlasticityRule {
public:
    virtual ~PlasticityRule() = default;

    /// One-time setup: allocate eligibility traces or auxiliary tensors
    /// inside the ConnectivityTensor (e.g., trace_pre, trace_post).
    virtual void initialize(ConnectivityTensor& conn) = 0;

    /// Apply the learning rule for one update step.
    ///
    /// @param conn       Connectivity tensor to modify (weights, masks).
    /// @param pre_state  State of the presynaptic population.
    /// @param post_state State of the postsynaptic population.
    /// @param t          Current simulation time.
    /// @param dt         Timestep duration.
    virtual void update(ConnectivityTensor& conn,
                        const State& pre_state,
                        const State& post_state,
                        Time t,
                        Duration dt) = 0;

    /// Reset all internal state (traces, accumulators) to initial values.
    virtual void reset() = 0;
};

} // namespace nrn
