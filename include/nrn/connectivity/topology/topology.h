#pragma once

#include <torch/torch.h>

#include <nrn/connectivity/connectivity_tensor.h>

namespace nrn {

/// Abstract base for topology generators.
///
/// A TopologyGenerator creates the block structure and initial mask
/// pattern for connectivity between a source population of size
/// `n_source` and a target population of size `n_target`.
///
/// Concrete generators (Random, DistanceDependent, SmallWorld, etc.)
/// implement the `generate()` method to produce a ConnectivityTensor
/// with:
///   - block_index filled with the CSR block structure
///   - weights initialized (e.g., random or uniform)
///   - structural_mask set according to the topology pattern
///   - modulatory_mask initialized to ones (no modulation)
///   - delays initialized to a default value
class TopologyGenerator {
public:
    virtual ~TopologyGenerator() = default;

    /// Generate connectivity between n_source and n_target neurons.
    ///
    /// @param n_source    Number of source (presynaptic) neurons.
    /// @param n_target    Number of target (postsynaptic) neurons.
    /// @param block_size  Side length B of each dense block.
    /// @param device      Device on which to allocate tensors.
    /// @return            Fully initialized ConnectivityTensor.
    virtual ConnectivityTensor generate(
        int64_t n_source,
        int64_t n_target,
        int64_t block_size,
        torch::Device device) = 0;
};

} // namespace nrn
