#pragma once

#include <nrn/core/types.h>
#include <nrn/core/state.h>

namespace nrn {

// Forward declaration -- avoid pulling in the full connectivity header.
struct ConnectivityTensor;

/// Ops table for synaptic plasticity rules.
///
/// PlasticityRule operates *on* connectivity tensors rather than maintaining
/// its own module state -- it is attached to a Connection and invoked by
/// the simulation loop at the appropriate timescale.
///
/// Concrete rules (STDP, BCM, homeostatic scaling, etc.) provide an ops
/// table with function pointers for initialize, update, and reset.
struct plasticity_ops {
    /// One-time setup: allocate eligibility traces or auxiliary tensors
    /// inside the ConnectivityTensor (e.g., trace_pre, trace_post).
    void (*initialize)(void* self, ConnectivityTensor& conn);

    /// Apply the learning rule for one update step.
    ///
    /// @param self       Opaque pointer to the concrete rule state.
    /// @param conn       Connectivity tensor to modify (weights, masks).
    /// @param pre_state  State of the presynaptic population.
    /// @param post_state State of the postsynaptic population.
    /// @param t          Current simulation time.
    /// @param dt         Timestep duration.
    void (*update)(void* self, ConnectivityTensor& conn,
                   const State& pre_state, const State& post_state,
                   double t, double dt);

    /// Reset all internal state (traces, accumulators) to initial values.
    void (*reset)(void* self);
};

/// Type-erased plasticity rule handle.
///
/// Holds an opaque pointer to the concrete rule state and a pointer to
/// the ops table that dispatches to the correct implementation.
struct PlasticityRule {
    void* impl;
    plasticity_ops* ops;
};

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

inline void plasticity_initialize(PlasticityRule* r, ConnectivityTensor& conn) {
    r->ops->initialize(r->impl, conn);
}

inline void plasticity_update(PlasticityRule* r, ConnectivityTensor& conn,
                              const State& pre, const State& post,
                              double t, double dt) {
    r->ops->update(r->impl, conn, pre, post, t, dt);
}

inline void plasticity_reset(PlasticityRule* r) {
    r->ops->reset(r->impl);
}

} // namespace nrn
