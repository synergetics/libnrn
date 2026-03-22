#include <nrn/graph/edge_ops.h>

#include <algorithm>

#include <nrn/connectivity/connectivity_tensor.h>
#include <nrn/plasticity/plasticity_rule.h>

namespace nrn {

// ---------------------------------------------------------------------------
// Block-sparse spike delivery (same algorithm as connection_deliver)
// ---------------------------------------------------------------------------

static void deliver_spike_blocksparse(ConnectivityTensor& ct,
                                      torch::Tensor& I_syn,
                                      const torch::Tensor& spikes) {
    const auto& bi = ct.block_index;
    int64_t B = ct.block_size;
    int64_t n_src = ct.n_source;
    int64_t n_tgt = ct.n_target;
    int64_t n_tgt_blocks = bi.n_rows();

    if (bi.n_blocks() == 0) return;

    auto W_eff = ct.effective_weights();

    auto row_ptr_cpu = bi.row_ptr.to(torch::kCPU);
    auto col_idx_cpu = bi.col_idx.to(torch::kCPU);
    auto rp = row_ptr_cpu.accessor<int32_t, 1>();
    auto ci = col_idx_cpu.accessor<int32_t, 1>();

    for (int64_t tr = 0; tr < n_tgt_blocks; ++tr) {
        int32_t block_start = rp[tr];
        int32_t block_end = rp[tr + 1];

        int64_t t_begin = tr * B;
        int64_t t_end = std::min(t_begin + B, n_tgt);
        int64_t t_size = t_end - t_begin;

        for (int32_t bi_idx = block_start; bi_idx < block_end; ++bi_idx) {
            int32_t sc = ci[bi_idx];
            int64_t s_begin = sc * B;
            int64_t s_end = std::min(s_begin + B, n_src);
            int64_t s_size = s_end - s_begin;

            auto W_block = W_eff[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);
            auto spike_block = spikes.slice(0, s_begin, s_end);
            auto I_block = torch::mv(W_block, spike_block);

            I_syn.slice(0, t_begin, t_end) += I_block;
        }
    }
}

// ---------------------------------------------------------------------------
// Dense spike delivery
// ---------------------------------------------------------------------------

static void deliver_spike_dense(const torch::Tensor& dense_weights,
                                torch::Tensor& I_syn,
                                const torch::Tensor& spikes) {
    // I_syn += W @ spikes — single matmul, perfect for tensor cores.
    I_syn.add_(torch::mv(dense_weights, spikes));
}

// ---------------------------------------------------------------------------
// Block-sparse continuous (gap junction) delivery
// ---------------------------------------------------------------------------

static void deliver_continuous_blocksparse(ConnectivityTensor& ct,
                                           torch::Tensor& I_syn_tgt,
                                           const torch::Tensor& V_src,
                                           const torch::Tensor& V_tgt) {
    const auto& bi = ct.block_index;
    int64_t B = ct.block_size;
    int64_t n_src = ct.n_source;
    int64_t n_tgt = ct.n_target;
    int64_t n_tgt_blocks = bi.n_rows();

    if (bi.n_blocks() == 0) return;

    auto W_eff = ct.effective_weights();

    auto row_ptr_cpu = bi.row_ptr.to(torch::kCPU);
    auto col_idx_cpu = bi.col_idx.to(torch::kCPU);
    auto rp = row_ptr_cpu.accessor<int32_t, 1>();
    auto ci = col_idx_cpu.accessor<int32_t, 1>();

    for (int64_t tr = 0; tr < n_tgt_blocks; ++tr) {
        int32_t block_start = rp[tr];
        int32_t block_end = rp[tr + 1];

        int64_t t_begin = tr * B;
        int64_t t_end = std::min(t_begin + B, n_tgt);
        int64_t t_size = t_end - t_begin;

        auto V_tgt_block = V_tgt.slice(0, t_begin, t_end);  // [t_size]

        for (int32_t bi_idx = block_start; bi_idx < block_end; ++bi_idx) {
            int32_t sc = ci[bi_idx];
            int64_t s_begin = sc * B;
            int64_t s_end = std::min(s_begin + B, n_src);
            int64_t s_size = s_end - s_begin;

            auto g_block = W_eff[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);
            auto V_src_block = V_src.slice(0, s_begin, s_end);  // [s_size]

            // dV[i] = sum_j g[i,j] * (V_src[j] - V_tgt[i])
            // = sum_j g[i,j] * V_src[j] - V_tgt[i] * sum_j g[i,j]
            auto I_from_src = torch::mv(g_block, V_src_block);  // [t_size]
            auto g_row_sum = g_block.sum(1);                     // [t_size]
            auto I_block = I_from_src - V_tgt_block * g_row_sum;

            I_syn_tgt.slice(0, t_begin, t_end) += I_block;
        }
    }
}

// ---------------------------------------------------------------------------
// Dense continuous (gap junction) delivery
// ---------------------------------------------------------------------------

static void deliver_continuous_dense(const torch::Tensor& dense_weights,
                                     torch::Tensor& I_syn_tgt,
                                     const torch::Tensor& V_src,
                                     const torch::Tensor& V_tgt) {
    // I_tgt[i] += sum_j g[i,j] * (V_src[j] - V_tgt[i])
    // = (g @ V_src) - V_tgt * (g.sum(dim=1))
    auto I_from_src = torch::mv(dense_weights, V_src);
    auto g_row_sum = dense_weights.sum(1);
    I_syn_tgt.add_(I_from_src - V_tgt * g_row_sum);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void edge_group_deliver_spike(EdgeGroup& eg,
                              NodeGroup& /*src*/,
                              NodeGroup& tgt,
                              const torch::Tensor& spikes,
                              double /*t*/, double /*dt*/) {
    if (!state_contains(*tgt.state, "I_syn")) return;
    auto I_syn = state_get(*tgt.state, "I_syn");

    if (eg.representation == EdgeRepresentation::Dense) {
        deliver_spike_dense(eg.dense_weights, I_syn, spikes);
    } else {
        TORCH_CHECK(eg.block_sparse != nullptr,
                    "edge_group_deliver_spike: block_sparse is null");
        deliver_spike_blocksparse(*eg.block_sparse, I_syn, spikes);
    }
}

void edge_group_deliver_continuous(EdgeGroup& eg,
                                   NodeGroup& src,
                                   NodeGroup& tgt,
                                   double /*t*/, double /*dt*/) {
    if (!state_contains(*tgt.state, "I_syn")) return;
    if (!state_contains(*src.state, "v")) return;
    if (!state_contains(*tgt.state, "v")) return;

    auto I_syn_tgt = state_get(*tgt.state, "I_syn");
    auto V_src = state_get(*src.state, "v");
    auto V_tgt = state_get(*tgt.state, "v");

    if (eg.representation == EdgeRepresentation::Dense) {
        deliver_continuous_dense(eg.dense_weights, I_syn_tgt, V_src, V_tgt);
    } else {
        TORCH_CHECK(eg.block_sparse != nullptr,
                    "edge_group_deliver_continuous: block_sparse is null");
        deliver_continuous_blocksparse(*eg.block_sparse, I_syn_tgt, V_src, V_tgt);
    }
}

void edge_group_update_plasticity(EdgeGroup& eg,
                                  const State& pre_state,
                                  const State& post_state,
                                  double t, double dt) {
    if (eg.representation == EdgeRepresentation::BlockSparse) {
        // Delegate to existing plasticity_update which operates on ConnectivityTensor.
        TORCH_CHECK(eg.block_sparse != nullptr,
                    "edge_group_update_plasticity: block_sparse is null");
        for (auto& rule : eg.plasticity_rules) {
            plasticity_update(&rule, *eg.block_sparse, pre_state, post_state, t, dt);
        }
    } else {
        // Dense representation: plasticity rules still operate on the block-sparse
        // ConnectivityTensor. For now, we skip plasticity on materialized-dense edges
        // since the dense_weights was a snapshot at compile time.
        // TODO(phase2): dense-native plasticity with dense_trace_pre/post.
    }
}

} // namespace nrn
