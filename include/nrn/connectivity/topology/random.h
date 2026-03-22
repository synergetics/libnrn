/// @file random.h
/// @brief Erdős–Rényi random (G(n,p)) topology generator.
///
/// @details
/// Every possible synapse exists independently with probability @c p.
/// All blocks in the CSR structure are allocated at Level 0 (full density
/// at block level); the structural mask within each block is sampled from
/// a Bernoulli distribution with probability @c probability.
///
/// @see RandomTopologyOptions, random_topology_create(), random_topology_as_generator()

#pragma once

#include <nrn/core/options.h>
#include <nrn/connectivity/topology/topology.h>

namespace nrn {

/// @brief Parameters for the Erdős–Rényi random topology generator.
struct RandomTopologyOptions {
    NRN_ARG(double, probability, 0.1);      ///< Connection probability (0–1).
    NRN_ARG(bool, allow_autapses, false);   ///< Whether a neuron may connect to itself.
};

/// @brief Internal state for the Erdős–Rényi random topology generator.
struct RandomTopology {
    RandomTopologyOptions opts; ///< Generator parameters.
};

/// @brief Allocate and return a new RandomTopology on the heap.
/// @param opts  Generator options.
/// @return Heap-allocated RandomTopology; caller must call @c random_topology_destroy().
RandomTopology* random_topology_create(const RandomTopologyOptions& opts = {});

/// @brief Free a RandomTopology.
/// @param r  Pointer to the generator; may be null (no-op).
void random_topology_destroy(RandomTopology* r);

/// @brief Generate a random ConnectivityTensor.
///
/// @param self        Opaque pointer to a RandomTopology.
/// @param n_source    Number of source neurons.
/// @param n_target    Number of target neurons.
/// @param block_size  Dense block side length.
/// @param device      Target device.
/// @return Fully initialised ConnectivityTensor.
ConnectivityTensor random_topology_generate(void* self, int64_t n_source,
                                            int64_t n_target,
                                            int64_t block_size,
                                            torch::Device device);

/// @brief Read-only access to generator options.
const RandomTopologyOptions& random_topology_options(const RandomTopology* r);

extern topology_ops random_topology_ops; ///< Ops table for RandomTopology.

/// @brief Wrap a RandomTopology into a type-erased TopologyGenerator.
/// @param r  Pointer to an existing RandomTopology.
/// @return TopologyGenerator handle.
inline TopologyGenerator random_topology_as_generator(RandomTopology* r) {
    return TopologyGenerator{static_cast<void*>(r), &random_topology_ops};
}

} // namespace nrn
