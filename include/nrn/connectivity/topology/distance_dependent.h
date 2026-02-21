#pragma once

#include <nrn/core/options.h>
#include <nrn/connectivity/topology/topology.h>

namespace nrn {

/// Options for distance-dependent connectivity.
struct DistanceDependentOptions {
    NRN_ARG(double, sigma, 200.0e-6);          ///< Gaussian width (metres).
    NRN_ARG(double, max_distance, 1000.0e-6);  ///< Cut-off distance (metres).
    NRN_ARG(double, min_probability, 0.0);      ///< Floor probability.
};

/// Internal state for the distance-dependent topology generator.
///
/// Connection probability decays as a Gaussian of the distance between
/// source and target neurons:
///
///     p(d) = max(exp(-d^2 / (2 * sigma^2)), min_probability)
///
/// Only blocks whose centres are within max_distance are allocated,
/// giving a spatially local block structure.  Neurons are assumed to
/// be arranged on a grid or supplied with explicit 3-D positions.
struct DistanceDepTopology {
    DistanceDependentOptions opts;
};

// ---------------------------------------------------------------------------
// Free functions operating on DistanceDepTopology
// ---------------------------------------------------------------------------

/// Allocate and return a new DistanceDepTopology on the heap.
DistanceDepTopology* distance_dep_topology_create(
    const DistanceDependentOptions& opts = {});

/// Free a DistanceDepTopology allocated with distance_dep_topology_create.
void distance_dep_topology_destroy(DistanceDepTopology* d);

/// Generate connectivity between n_source and n_target neurons.
ConnectivityTensor distance_dep_topology_generate(void* self,
                                                   int64_t n_source,
                                                   int64_t n_target,
                                                   int64_t block_size,
                                                   torch::Device device);

/// Access options (read-only).
const DistanceDependentOptions& distance_dep_topology_options(
    const DistanceDepTopology* d);

/// Ops table for DistanceDepTopology.
extern topology_ops distance_dep_topology_ops;

/// Wrap a DistanceDepTopology pointer into a type-erased TopologyGenerator handle.
inline TopologyGenerator distance_dep_topology_as_generator(
    DistanceDepTopology* d) {
    return TopologyGenerator{static_cast<void*>(d), &distance_dep_topology_ops};
}

} // namespace nrn
