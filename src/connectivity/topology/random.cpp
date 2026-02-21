#include <nrn/connectivity/topology/random.h>

#include <cmath>

namespace nrn {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

RandomTopology* random_topology_create(const RandomTopologyOptions& opts) {
    auto* r = new RandomTopology{};
    r->opts = opts;
    return r;
}

void random_topology_destroy(RandomTopology* r) {
    delete r;
}

// ---------------------------------------------------------------------------
// Generate implementation (void* self -> RandomTopology*)
// ---------------------------------------------------------------------------

ConnectivityTensor random_topology_generate(void* self, int64_t n_source,
                                            int64_t n_target,
                                            int64_t block_size,
                                            torch::Device device) {
    auto* r = static_cast<RandomTopology*>(self);

    // Compute number of blocks in each dimension (ceil division).
    int64_t n_src_blocks = (n_source + block_size - 1) / block_size;
    int64_t n_tgt_blocks = (n_target + block_size - 1) / block_size;

    // For Erdos-Renyi, all blocks exist (full block structure).
    int64_t n_blocks = n_src_blocks * n_tgt_blocks;

    // Build CSR block index: every target block row has n_src_blocks entries.
    auto row_ptr = torch::zeros({n_tgt_blocks + 1}, torch::kInt32);
    auto col_idx = torch::zeros({n_blocks}, torch::kInt32);

    auto rp = row_ptr.accessor<int32_t, 1>();
    auto ci = col_idx.accessor<int32_t, 1>();

    int32_t offset = 0;
    for (int64_t row = 0; row < n_tgt_blocks; ++row) {
        rp[row] = offset;
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

    // Structural mask: Bernoulli with probability p.
    ct.structural_mask = torch::bernoulli(
        torch::full({n_blocks, block_size, block_size},
                    r->opts.probability(), topts));

    // Remove autapses if same population (diagonal blocks, i==j).
    if (!r->opts.allow_autapses() && n_source == n_target) {
        for (int64_t b = 0; b < std::min(n_src_blocks, n_tgt_blocks); ++b) {
            int64_t block_idx = b * n_src_blocks + b;
            if (block_idx < n_blocks) {
                // Zero the diagonal of this block.
                auto block_mask = ct.structural_mask[block_idx];
                auto diag_size = std::min(block_size, block_size);
                for (int64_t i = 0; i < diag_size; ++i) {
                    block_mask[i][i] = 0.0f;
                }
            }
        }
    }

    // Modulatory mask: all ones (no modulation).
    ct.modulatory_mask = torch::ones({n_blocks, block_size, block_size}, topts);

    // Delays: all ones (minimum delay of 1 step).
    ct.delays = torch::ones({n_blocks, block_size, block_size},
                            torch::TensorOptions().dtype(torch::kInt32).device(device));

    return ct;
}

// ---------------------------------------------------------------------------
// Read-only accessor
// ---------------------------------------------------------------------------

const RandomTopologyOptions& random_topology_options(const RandomTopology* r) {
    return r->opts;
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

topology_ops random_topology_ops = {
    random_topology_generate,
};

} // namespace nrn
