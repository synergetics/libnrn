#pragma once

#include <nrn/core/options.h>
#include <nrn/connectivity/topology/topology.h>

namespace nrn {

/// Options for Erdos-Renyi random connectivity.
struct RandomTopologyOptions {
    NRN_ARG(double, probability, 0.1);
    NRN_ARG(bool, allow_autapses, false);
};

/// Erdos-Renyi random topology generator.
///
/// Every possible connection exists independently with probability `p`.
/// All blocks in the block structure are allocated (full density at
/// block level); the structural mask within each block is random
/// Bernoulli with the specified probability.
class Random : public TopologyGenerator {
public:
    explicit Random(const RandomTopologyOptions& opts = {});

    ConnectivityTensor generate(
        int64_t n_source,
        int64_t n_target,
        int64_t block_size,
        torch::Device device) override;

private:
    RandomTopologyOptions opts_;
};

} // namespace nrn
