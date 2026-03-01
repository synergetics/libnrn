#include <nrn/connectivity/topology/distance_dependent.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nrn {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

DistanceDepTopology* distance_dep_topology_create(
    const DistanceDependentOptions& opts) {
    auto* d = new DistanceDepTopology{};
    d->opts = opts;
    return d;
}

void distance_dep_topology_destroy(DistanceDepTopology* d) {
    delete d;
}

// ---------------------------------------------------------------------------
// Generate implementation (void* self -> DistanceDepTopology*)
// ---------------------------------------------------------------------------

ConnectivityTensor distance_dep_topology_generate(void* self,
                                                   int64_t n_source,
                                                   int64_t n_target,
                                                   int64_t block_size,
                                                   torch::Device device) {
    auto* d = static_cast<DistanceDepTopology*>(self);
    double sigma = d->opts.sigma();
    double max_dist = d->opts.max_distance();
    double min_prob = d->opts.min_probability();
    double two_sigma_sq = 2.0 * sigma * sigma;

    int64_t n_src_blocks = (n_source + block_size - 1) / block_size;
    int64_t n_tgt_blocks = (n_target + block_size - 1) / block_size;

    // Assign uniform 1D positions in [0, max_distance) to neurons.
    // Block centres determine which blocks to allocate.
    // Source positions: evenly spaced.
    // Target positions: evenly spaced.
    auto src_spacing = (n_source > 1) ? max_dist / static_cast<double>(n_source) : 0.0;
    auto tgt_spacing = (n_target > 1) ? max_dist / static_cast<double>(n_target) : 0.0;

    // Determine which blocks to allocate: only those whose block centres
    // are within max_distance of each other.
    std::vector<int32_t> row_ptr_vec(n_tgt_blocks + 1, 0);
    std::vector<int32_t> col_idx_vec;

    for (int64_t tr = 0; tr < n_tgt_blocks; ++tr) {
        row_ptr_vec[tr] = static_cast<int32_t>(col_idx_vec.size());
        double tgt_center = (tr * block_size + std::min((tr + 1) * block_size, n_target)) * 0.5 * tgt_spacing;

        for (int64_t sc = 0; sc < n_src_blocks; ++sc) {
            double src_center = (sc * block_size + std::min((sc + 1) * block_size, n_source)) * 0.5 * src_spacing;
            double block_dist = std::abs(tgt_center - src_center);

            // Check if any neuron pair in this block could have non-trivial probability.
            double max_prob_in_block = std::exp(-block_dist * block_dist / two_sigma_sq);
            if (max_prob_in_block > 0.01 || min_prob > 0.01) {
                col_idx_vec.push_back(static_cast<int32_t>(sc));
            }
        }
    }
    row_ptr_vec[n_tgt_blocks] = static_cast<int32_t>(col_idx_vec.size());

    int64_t n_blocks = static_cast<int64_t>(col_idx_vec.size());

    // Build CSR tensors.
    auto row_ptr = torch::zeros({n_tgt_blocks + 1}, torch::kInt32);
    auto col_idx = torch::zeros({std::max(n_blocks, int64_t(1))}, torch::kInt32);

    auto rp = row_ptr.accessor<int32_t, 1>();
    for (int64_t i = 0; i <= n_tgt_blocks; ++i) {
        rp[i] = row_ptr_vec[i];
    }
    if (n_blocks > 0) {
        auto ci = col_idx.accessor<int32_t, 1>();
        for (int64_t i = 0; i < n_blocks; ++i) {
            ci[i] = col_idx_vec[i];
        }
    }

    ConnectivityTensor ct;
    ct.block_index.row_ptr = row_ptr.to(device);
    ct.block_index.col_idx = col_idx.to(device);
    ct.block_size = block_size;
    ct.n_source = n_source;
    ct.n_target = n_target;

    auto topts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Weights: random uniform [0, 1).
    ct.weights = torch::rand({n_blocks, block_size, block_size}, topts);

    // Structural mask: distance-dependent Gaussian probability per synapse.
    ct.structural_mask = torch::zeros({n_blocks, block_size, block_size}, topts);

    int64_t block_idx = 0;
    for (int64_t tr = 0; tr < n_tgt_blocks; ++tr) {
        int32_t blk_start = row_ptr_vec[tr];
        int32_t blk_end   = row_ptr_vec[tr + 1];

        int64_t t_begin = tr * block_size;
        int64_t t_end   = std::min(t_begin + block_size, n_target);

        for (int32_t bi = blk_start; bi < blk_end; ++bi) {
            int32_t sc = col_idx_vec[bi];
            int64_t s_begin = sc * block_size;
            int64_t s_end   = std::min(s_begin + block_size, n_source);

            // Compute per-synapse probability tensor for this block.
            auto prob_block = torch::zeros({block_size, block_size}, topts);
            auto pb = prob_block.accessor<float, 2>();

            for (int64_t ti = 0; ti < t_end - t_begin; ++ti) {
                double tgt_pos = (t_begin + ti) * tgt_spacing;
                for (int64_t si = 0; si < s_end - s_begin; ++si) {
                    double src_pos = (s_begin + si) * src_spacing;
                    double dist = std::abs(tgt_pos - src_pos);
                    double p = std::exp(-dist * dist / two_sigma_sq);
                    p = std::max(p, min_prob);
                    pb[ti][si] = static_cast<float>(p);
                }
            }

            ct.structural_mask[bi] = torch::bernoulli(prob_block);
            ++block_idx;
        }
    }

    // Modulatory mask: all ones.
    ct.modulatory_mask = torch::ones({n_blocks, block_size, block_size}, topts);

    // Delays: all ones.
    ct.delays = torch::ones({n_blocks, block_size, block_size},
                            torch::TensorOptions().dtype(torch::kInt32).device(device));

    return ct;
}

// ---------------------------------------------------------------------------
// Read-only accessor
// ---------------------------------------------------------------------------

const DistanceDependentOptions& distance_dep_topology_options(
    const DistanceDepTopology* d) {
    return d->opts;
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

topology_ops distance_dep_topology_ops = {
    distance_dep_topology_generate,
};

} // namespace nrn
