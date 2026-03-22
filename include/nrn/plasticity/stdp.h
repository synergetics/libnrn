/// @file stdp.h
/// @brief Pair-based Spike-Timing-Dependent Plasticity (STDP) rule.
///
/// @details
/// Implements the classic asymmetric STDP window via eligibility traces
/// stored inside the ConnectivityTensor:
/// @code
///   dw = a_plus  * pre_trace  * post_spike
///      - a_minus * post_trace * pre_spike
///   w += learning_rate * dw * structural_mask * modulatory_mask
///   w  = clamp(w, w_min, w_max)
///
///   d(trace_pre)  / dt = -trace_pre  / tau_plus
///   d(trace_post) / dt = -trace_post / tau_minus
///   on pre  spike: trace_pre  += 1
///   on post spike: trace_post += 1
/// @endcode
///
/// @see STDPOptions, stdp_create(), stdp_as_rule()

#pragma once

#include <nrn/plasticity/plasticity_rule.h>
#include <nrn/plasticity/options.h>

namespace nrn {

// Forward declaration.
struct ConnectivityTensor;

/// @brief Internal state for the STDP plasticity rule.
///
/// @details
/// Decay factors @c decay_pre and @c decay_post are cached and recomputed
/// whenever @c dt changes, avoiding redundant @c exp() evaluations.
struct STDPState {
    STDPOptions opts; ///< Rule parameters.

    double cached_dt  = 0.0; ///< @c dt value used to compute the cached decay factors.
    double decay_pre  = 0.0; ///< Pre-trace per-step decay: @c exp(-dt / tau_plus).
    double decay_post = 0.0; ///< Post-trace per-step decay: @c exp(-dt / tau_minus).
};

/// @brief Allocate and initialise an STDP rule on the heap.
/// @param opts  Rule parameters.
/// @return Heap-allocated STDPState; caller must call @c stdp_destroy().
STDPState* stdp_create(const STDPOptions& opts = {});

/// @brief Free an STDPState.
/// @param s  Pointer to the state; may be null (no-op).
void stdp_destroy(STDPState* s);

/// @brief Allocate @c trace_pre and @c trace_post inside @p conn.
///
/// @param self  Opaque pointer to an STDPState.
/// @param conn  ConnectivityTensor to extend; traces match the weight shape.
void stdp_initialize(void* self, ConnectivityTensor& conn);

/// @brief Apply one STDP weight update step.
///
/// @param self       Opaque pointer to an STDPState.
/// @param conn       ConnectivityTensor whose weights are modified in place.
/// @param pre_state  State of the presynaptic population (provides @c "spike").
/// @param post_state State of the postsynaptic population (provides @c "spike").
/// @param t          Current simulation time (s).
/// @param dt         Timestep duration (s).
void stdp_update(void* self, ConnectivityTensor& conn,
                 const State& pre_state, const State& post_state,
                 double t, double dt);

/// @brief Zero all eligibility traces and reset cached decay factors.
/// @param self  Opaque pointer to an STDPState.
void stdp_reset(void* self);

/// @brief Read-only access to STDP parameters.
const STDPOptions& stdp_options(const STDPState* s);

extern plasticity_ops stdp_ops; ///< Ops table for STDP.

/// @brief Wrap an STDPState into a type-erased PlasticityRule handle.
/// @param s  Pointer to an existing STDPState.
/// @return PlasticityRule with @c stdp_ops dispatch table.
inline PlasticityRule stdp_as_rule(STDPState* s) {
    return PlasticityRule{static_cast<void*>(s), &stdp_ops};
}

} // namespace nrn
