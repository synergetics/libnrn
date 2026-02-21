#pragma once

#include <torch/torch.h>

#include <nrn/connectivity/connectivity_tensor.h>

namespace nrn {

/// Ops table for topology generators.
///
/// A topology generator creates the block structure and initial mask
/// pattern for connectivity between a source population of size
/// n_source and a target population of size n_target.
///
/// Concrete generators (Random, DistanceDependent, SmallWorld, etc.)
/// provide an ops table with a generate function that produces a
/// ConnectivityTensor with:
///   - block_index filled with the CSR block structure
///   - weights initialized (e.g., random or uniform)
///   - structural_mask set according to the topology pattern
///   - modulatory_mask initialized to ones (no modulation)
///   - delays initialized to a default value
struct topology_ops {
    /// Generate connectivity between n_source and n_target neurons.
    ///
    /// @param self        Opaque pointer to the concrete generator state.
    /// @param n_source    Number of source (presynaptic) neurons.
    /// @param n_target    Number of target (postsynaptic) neurons.
    /// @param block_size  Side length B of each dense block.
    /// @param device      Device on which to allocate tensors.
    /// @return            Fully initialized ConnectivityTensor.
    ConnectivityTensor (*generate)(void* self, int64_t n_source,
                                   int64_t n_target, int64_t block_size,
                                   torch::Device device);
};

/// Type-erased topology generator handle.
///
/// Holds an opaque pointer to the concrete generator state and a pointer
/// to the ops table that dispatches to the correct implementation.
struct TopologyGenerator {
    void* impl;
    topology_ops* ops;
};

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

inline ConnectivityTensor topology_generate(TopologyGenerator* g,
                                            int64_t n_src, int64_t n_tgt,
                                            int64_t block_size,
                                            torch::Device device) {
    return g->ops->generate(g->impl, n_src, n_tgt, block_size, device);
}

} // namespace nrn
