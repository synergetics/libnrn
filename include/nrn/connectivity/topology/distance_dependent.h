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

/// Distance-dependent topology generator.
///
/// Connection probability decays as a Gaussian of the distance between
/// source and target neurons:
///
///     p(d) = max(exp(-d^2 / (2 * sigma^2)), min_probability)
///
/// Only blocks whose centres are within `max_distance` are allocated,
/// giving a spatially local block structure.  Neurons are assumed to
/// be arranged on a grid or supplied with explicit 3-D positions.
class DistanceDependent : public TopologyGenerator {
public:
    explicit DistanceDependent(const DistanceDependentOptions& opts = {});

    ConnectivityTensor generate(
        int64_t n_source,
        int64_t n_target,
        int64_t block_size,
        torch::Device device) override;

private:
    DistanceDependentOptions opts_;
};

} // namespace nrn
