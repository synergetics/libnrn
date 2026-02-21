#pragma once

#include <nrn/core/options.h>
#include <nrn/connectivity/topology/topology.h>

namespace nrn {

/// Options for Erdos-Renyi random connectivity.
struct RandomTopologyOptions {
    NRN_ARG(double, probability, 0.1);
    NRN_ARG(bool, allow_autapses, false);
};

/// Internal state for the Erdos-Renyi random topology generator.
///
/// Every possible connection exists independently with probability p.
/// All blocks in the block structure are allocated (full density at
/// block level); the structural mask within each block is random
/// Bernoulli with the specified probability.
struct RandomTopology {
    RandomTopologyOptions opts;
};

// ---------------------------------------------------------------------------
// Free functions operating on RandomTopology
// ---------------------------------------------------------------------------

/// Allocate and return a new RandomTopology on the heap.
RandomTopology* random_topology_create(const RandomTopologyOptions& opts = {});

/// Free a RandomTopology allocated with random_topology_create.
void random_topology_destroy(RandomTopology* r);

/// Generate connectivity between n_source and n_target neurons.
ConnectivityTensor random_topology_generate(void* self, int64_t n_source,
                                            int64_t n_target,
                                            int64_t block_size,
                                            torch::Device device);

/// Access options (read-only).
const RandomTopologyOptions& random_topology_options(const RandomTopology* r);

/// Ops table for RandomTopology.
extern topology_ops random_topology_ops;

/// Wrap a RandomTopology pointer into a type-erased TopologyGenerator handle.
inline TopologyGenerator random_topology_as_generator(RandomTopology* r) {
    return TopologyGenerator{static_cast<void*>(r), &random_topology_ops};
}

} // namespace nrn
