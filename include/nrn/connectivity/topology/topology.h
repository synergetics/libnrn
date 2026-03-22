/// @file topology.h
/// @brief Type-erased TopologyGenerator handle and ops table.
///
/// @details
/// A TopologyGenerator creates the block structure and initial connectivity
/// pattern (weights + structural mask) for a Connection.  Concrete
/// generators (RandomTopology, DistanceDepTopology, etc.) each provide an
/// ops table and wrap themselves via a @c *_as_generator() function.
///
/// @see RandomTopology, DistanceDepTopology, topology_generate()

#pragma once

#include <torch/torch.h>

#include <nrn/connectivity/connectivity_tensor.h>

namespace nrn {

/// @brief Ops table for topology generators.
///
/// @details
/// Concrete generators provide a static instance of this struct.  The
/// single @c generate function pointer produces a fully initialised
/// ConnectivityTensor with:
///   - block_index filled with the CSR block structure.
///   - weights initialised (e.g. random or uniform).
///   - structural_mask set according to the topology pattern.
///   - modulatory_mask initialised to @c 1.0 (no modulation).
///   - delays initialised to a default value.
struct topology_ops {
    /// @brief Generate connectivity between @p n_source and @p n_target neurons.
    ///
    /// @param self        Opaque pointer to the concrete generator state.
    /// @param n_source    Number of source (presynaptic) neurons.
    /// @param n_target    Number of target (postsynaptic) neurons.
    /// @param block_size  Side length @c B of each dense block.
    /// @param device      Device on which to allocate tensors.
    /// @return Fully initialised ConnectivityTensor.
    ConnectivityTensor (*generate)(void* self, int64_t n_source,
                                   int64_t n_target, int64_t block_size,
                                   torch::Device device);
};

/// @brief Type-erased topology generator handle.
///
/// @details
/// Holds an opaque @c void* to the concrete generator and a pointer to
/// its ops table.  Wrap concrete generators with @c *_as_generator().
///
/// @see random_topology_as_generator(), distance_dep_topology_as_generator()
struct TopologyGenerator {
    void* impl;           ///< Opaque pointer to the concrete generator struct.
    topology_ops* ops;    ///< Dispatch table.
};

/// @brief Dispatch helper: generate a ConnectivityTensor from a TopologyGenerator.
///
/// @param g          Topology generator; must not be null.
/// @param n_src      Number of source neurons.
/// @param n_tgt      Number of target neurons.
/// @param block_size Dense block side length.
/// @param device     Target device for tensor allocation.
/// @return Fully initialised ConnectivityTensor.
inline ConnectivityTensor topology_generate(TopologyGenerator* g,
                                            int64_t n_src, int64_t n_tgt,
                                            int64_t block_size,
                                            torch::Device device) {
    return g->ops->generate(g->impl, n_src, n_tgt, block_size, device);
}

} // namespace nrn
