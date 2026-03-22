/// @file edge_ops.h
/// @brief Per-EdgeGroup operations invoked during each phase of graph_step().
///
/// @details
/// These free functions form the inner loop of the simulation.  They are
/// called by graph_step() once per EdgeGroup per timestep and dispatch to
/// either the dense or block-sparse code path based on
/// EdgeGroup::representation.
///
/// @see graph_step(), EdgeGroup, EdgeTrigger, EdgeRepresentation

#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/graph/graph.h>

namespace nrn {

/// @brief Deliver spike-triggered synaptic current through an EdgeGroup (Phase 2).
///
/// @details
/// Reads the binary @c spikes tensor for @p src and accumulates weighted
/// input current into the @c I_syn variable of @p tgt:
/// @code
///   // Dense path:
///   I_syn[tgt] += W @ spikes[src]
///
///   // BlockSparse path:
///   for each block b:  I_syn[tgt_b] += W_b @ spikes[src_b]
/// @endcode
/// This function is a no-op if @p eg has EdgeTrigger::Continuous.
///
/// @param eg      EdgeGroup defining connectivity, weights, and storage format.
/// @param src     Source NodeGroup whose spike tensor drives the delivery.
/// @param tgt     Target NodeGroup whose @c I_syn state variable is updated.
/// @param spikes  Binary float spike tensor for @p src (@c 1.0 = fired).  May
///                be sourced from a delay line rather than live state.
/// @param t       Current simulation time in seconds, forwarded to the synapse model.
/// @param dt      Timestep duration in seconds.
///
/// @see edge_group_deliver_continuous(), EdgeTrigger::Spike
void edge_group_deliver_spike(EdgeGroup& eg,
                              NodeGroup& src,
                              NodeGroup& tgt,
                              const torch::Tensor& spikes,
                              double t, double dt);

/// @brief Deliver continuous (gap-junction) current through an EdgeGroup (Phase 2).
///
/// @details
/// Computes voltage-difference coupling current every timestep and accumulates
/// it into @c I_syn of @p tgt:
/// @code
///   // Dense path (avoids per-element loop):
///   I_tgt += W @ V_src - rowsum(W) * V_tgt
///
///   // BlockSparse path:
///   for each block b:  I_tgt_b += W_b @ V_src_b - rowsum(W_b) * V_tgt_b
/// @endcode
/// Net current is conservative: total injected current sums to zero across
/// the coupled populations.
///
/// This function is a no-op if @p eg has EdgeTrigger::Spike.
///
/// @param eg   EdgeGroup defining coupling conductances and storage format.
/// @param src  Source NodeGroup providing the @c V (membrane potential) tensor.
/// @param tgt  Target NodeGroup whose @c I_syn state variable is updated.
/// @param t    Current simulation time in seconds.
/// @param dt   Timestep duration in seconds.
///
/// @see edge_group_deliver_spike(), EdgeTrigger::Continuous
void edge_group_deliver_continuous(EdgeGroup& eg,
                                   NodeGroup& src,
                                   NodeGroup& tgt,
                                   double t, double dt);

/// @brief Apply all plasticity rules attached to an EdgeGroup (Phase 5).
///
/// @details
/// Iterates over EdgeGroup::plasticity_rules and updates synaptic weights
/// based on pre- and post-synaptic activity in @p pre_state and @p post_state:
///   - **Dense path**: operates on whole-matrix trace tensors
///     (EdgeGroup::dense_trace_pre, EdgeGroup::dense_trace_post).
///   - **BlockSparse path**: delegates to @c plasticity_update() per rule,
///     using the ConnectivityTensor block structure.
///
/// @param eg          EdgeGroup whose plasticity rules are evaluated.
/// @param pre_state   State of the presynaptic NodeGroup (provides @c spikes and traces).
/// @param post_state  State of the postsynaptic NodeGroup (provides @c spikes and traces).
/// @param t           Current simulation time in seconds.
/// @param dt          Timestep duration in seconds.
///
/// @see PlasticityRule, EdgeGroup::plasticity_rules
void edge_group_update_plasticity(EdgeGroup& eg,
                                  const State& pre_state,
                                  const State& post_state,
                                  double t, double dt);

} // namespace nrn
