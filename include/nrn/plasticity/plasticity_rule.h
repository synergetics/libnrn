/// @file plasticity_rule.h
/// @brief Type-erased PlasticityRule handle and plasticity_ops dispatch table.
///
/// @details
/// PlasticityRule operates *on* ConnectivityTensor objects rather than
/// maintaining independent module state.  A rule is attached to a Connection
/// and invoked by the simulation loop at the appropriate timescale (Phase 5
/// of graph_step()).
///
/// Concrete rules (STDP, BCM, homeostatic scaling, …) provide a static
/// @c plasticity_ops table and are wrapped with @c *_as_rule().
///
/// @see STDPState, stdp_as_rule(), PlasticityRule

#pragma once

#include <nrn/core/types.h>
#include <nrn/core/state.h>

namespace nrn {

// Forward declaration.
struct ConnectivityTensor;

/// @brief Ops table for synaptic plasticity rules.
///
/// @details
/// Each concrete rule provides a static instance of this struct.
/// The three operations cover the full lifecycle: one-time setup,
/// per-step update, and reset.
struct plasticity_ops {
    /// @brief One-time initialisation: allocate eligibility traces in @p conn.
    ///
    /// @param self  Opaque pointer to the concrete rule state.
    /// @param conn  ConnectivityTensor to extend with trace tensors.
    void (*initialize)(void* self, ConnectivityTensor& conn);

    /// @brief Apply the learning rule for one simulation step.
    ///
    /// @param self       Opaque pointer to the concrete rule state.
    /// @param conn       ConnectivityTensor whose weights are updated in place.
    /// @param pre_state  State of the presynaptic population.
    /// @param post_state State of the postsynaptic population.
    /// @param t          Current simulation time (s).
    /// @param dt         Timestep duration (s).
    void (*update)(void* self, ConnectivityTensor& conn,
                   const State& pre_state, const State& post_state,
                   double t, double dt);

    /// @brief Reset all internal traces and accumulators to their initial values.
    ///
    /// @param self  Opaque pointer to the concrete rule state.
    void (*reset)(void* self);
};

/// @brief Type-erased handle to any plasticity rule.
///
/// @details
/// Holds an opaque @c void* to the concrete rule state and a pointer to
/// its ops table.  Wrap concrete rules with e.g. @c stdp_as_rule().
///
/// @see plasticity_ops, stdp_as_rule()
struct PlasticityRule {
    void* impl;           ///< Opaque pointer to the concrete rule struct.
    plasticity_ops* ops;  ///< Dispatch table.
};

/// @brief Initialise a plasticity rule on a ConnectivityTensor.
/// @see plasticity_ops::initialize
inline void plasticity_initialize(PlasticityRule* r, ConnectivityTensor& conn) {
    r->ops->initialize(r->impl, conn);
}

/// @brief Apply one plasticity update step.
/// @see plasticity_ops::update
inline void plasticity_update(PlasticityRule* r, ConnectivityTensor& conn,
                              const State& pre, const State& post,
                              double t, double dt) {
    r->ops->update(r->impl, conn, pre, post, t, dt);
}

/// @brief Reset all rule state (traces, accumulators).
/// @see plasticity_ops::reset
inline void plasticity_reset(PlasticityRule* r) {
    r->ops->reset(r->impl);
}

} // namespace nrn
