/// @file distance_dependent.h
/// @brief Gaussian distance-dependent topology generator.
///
/// @details
/// Connection probability decays as a Gaussian of inter-neuron distance:
/// @code
///   p(d) = max(exp(-d^2 / (2 * sigma^2)), min_probability)
/// @endcode
/// Only blocks whose centres are within @c max_distance are allocated in
/// the CSR structure, giving a spatially local block topology.  Neurons
/// must be arranged on a grid or provided with explicit 3-D positions via
/// population_set_positions().
///
/// @see DistanceDependentOptions, distance_dep_topology_create(),
///      distance_dep_topology_as_generator()

#pragma once

#include <nrn/core/options.h>
#include <nrn/connectivity/topology/topology.h>

namespace nrn {

/// @brief Parameters for the Gaussian distance-dependent topology generator.
struct DistanceDependentOptions {
    NRN_ARG(double, sigma, 200.0e-6);          ///< Gaussian width @sigma (metres).
    NRN_ARG(double, max_distance, 1000.0e-6);  ///< Connection cut-off distance (metres).
    NRN_ARG(double, min_probability, 0.0);      ///< Floor probability at any distance.
};

/// @brief Internal state for the distance-dependent topology generator.
struct DistanceDepTopology {
    DistanceDependentOptions opts; ///< Generator parameters.
};

/// @brief Allocate and return a new DistanceDepTopology on the heap.
/// @param opts  Generator options.
/// @return Heap-allocated DistanceDepTopology; caller must call @c distance_dep_topology_destroy().
DistanceDepTopology* distance_dep_topology_create(
    const DistanceDependentOptions& opts = {});

/// @brief Free a DistanceDepTopology.
/// @param d  Pointer to the generator; may be null (no-op).
void distance_dep_topology_destroy(DistanceDepTopology* d);

/// @brief Generate a distance-dependent ConnectivityTensor.
///
/// @param self        Opaque pointer to a DistanceDepTopology.
/// @param n_source    Number of source neurons.
/// @param n_target    Number of target neurons.
/// @param block_size  Dense block side length.
/// @param device      Target device.
/// @return Fully initialised ConnectivityTensor.
ConnectivityTensor distance_dep_topology_generate(void* self,
                                                   int64_t n_source,
                                                   int64_t n_target,
                                                   int64_t block_size,
                                                   torch::Device device);

/// @brief Read-only access to generator options.
const DistanceDependentOptions& distance_dep_topology_options(
    const DistanceDepTopology* d);

extern topology_ops distance_dep_topology_ops; ///< Ops table for DistanceDepTopology.

/// @brief Wrap a DistanceDepTopology into a type-erased TopologyGenerator.
/// @param d  Pointer to an existing DistanceDepTopology.
/// @return TopologyGenerator handle.
inline TopologyGenerator distance_dep_topology_as_generator(
    DistanceDepTopology* d) {
    return TopologyGenerator{static_cast<void*>(d), &distance_dep_topology_ops};
}

} // namespace nrn
