#pragma once

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/graph/graph.h>

namespace nrn {

/// Deliver spike-triggered synaptic current through an edge group.
///
/// For Dense: I_syn += dense_weights @ spikes
/// For BlockSparse: CSR block iteration with torch::mv per block.
void edge_group_deliver_spike(EdgeGroup& eg,
                              NodeGroup& src,
                              NodeGroup& tgt,
                              const torch::Tensor& spikes,
                              double t, double dt);

/// Deliver continuous (gap-junction) current through an edge group.
///
/// For Dense: I_tgt += (g * (V_pre - V_post)).sum(dim=1)
/// For BlockSparse: CSR block iteration with per-block coupling.
void edge_group_deliver_continuous(EdgeGroup& eg,
                                   NodeGroup& src,
                                   NodeGroup& tgt,
                                   double t, double dt);

/// Run all plasticity rules attached to an edge group.
///
/// For Dense: whole-matrix trace operations (no block iteration).
/// For BlockSparse: delegates to existing plasticity_update() per rule.
void edge_group_update_plasticity(EdgeGroup& eg,
                                  const State& pre_state,
                                  const State& post_state,
                                  double t, double dt);

} // namespace nrn
