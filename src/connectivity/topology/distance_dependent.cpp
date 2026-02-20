#include <nrn/connectivity/topology/distance_dependent.h>

#include <cmath>

namespace nrn {

DistanceDependent::DistanceDependent(const DistanceDependentOptions& opts)
    : opts_(opts) {}

ConnectivityTensor DistanceDependent::generate(
    int64_t n_source,
    int64_t n_target,
    int64_t block_size,
    torch::Device device) {

    // Compute number of blocks in each dimension.
    int64_t n_src_blocks = (n_source + block_size - 1) / block_size;
    int64_t n_tgt_blocks = (n_target + block_size - 1) / block_size;

    // For this minimal implementation, allocate all blocks (like Random).
    // A full implementation would only allocate blocks whose centres
    // are within max_distance.
    int64_t n_blocks = n_src_blocks * n_tgt_blocks;

    // Build CSR block index.
    auto row_ptr = torch::zeros({n_tgt_blocks + 1}, torch::kInt32);
    auto col_idx = torch::zeros({n_blocks}, torch::kInt32);

    auto rp = row_ptr.accessor<int32_t, 1>();
    auto ci = col_idx.accessor<int32_t, 1>();

    int32_t offset = 0;
    for (int64_t r = 0; r < n_tgt_blocks; ++r) {
        rp[r] = offset;
        for (int64_t c = 0; c < n_src_blocks; ++c) {
            ci[offset] = static_cast<int32_t>(c);
            ++offset;
        }
    }
    rp[n_tgt_blocks] = offset;

    ConnectivityTensor ct;
    ct.block_index.row_ptr = row_ptr.to(device);
    ct.block_index.col_idx = col_idx.to(device);
    ct.block_size = block_size;
    ct.n_source = n_source;
    ct.n_target = n_target;

    auto topts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Weights: random uniform [0, 1).
    ct.weights = torch::rand({n_blocks, block_size, block_size}, topts);

    // Structural mask: for now, use uniform probability based on sigma.
    // A full implementation would compute per-synapse distances.
    double p = std::exp(-0.5); // ~0.6 as a placeholder
    ct.structural_mask = torch::bernoulli(
        torch::full({n_blocks, block_size, block_size}, p, topts));

    // Modulatory mask: all ones.
    ct.modulatory_mask = torch::ones({n_blocks, block_size, block_size}, topts);

    // Delays: all ones.
    ct.delays = torch::ones({n_blocks, block_size, block_size},
                            torch::TensorOptions().dtype(torch::kInt32).device(device));

    return ct;
}

} // namespace nrn
