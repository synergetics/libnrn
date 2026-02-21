#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/topology.h>
#include <nrn/network/population.h>
#include <nrn/plasticity/plasticity_rule.h>

#include <algorithm>

namespace nrn {

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

Connection::Connection(std::shared_ptr<Population> source,
                       std::shared_ptr<Population> target,
                       ConnectivityTensor connectivity,
                       std::shared_ptr<torch::nn::Module> synapse,
                       ConnectOptions options)
    : source_(std::move(source)),
      target_(std::move(target)),
      connectivity_(std::move(connectivity)),
      synapse_(std::move(synapse)),
      options_(std::move(options)) {}

void Connection::deliver(const torch::Tensor& spikes,
                         Time /*t*/, Duration /*dt*/) {
    // Get target population's I_syn via the State bag.
    auto& target_state = target_->state();
    if (!target_state.contains("I_syn")) {
        return;  // Target state not yet initialized — skip.
    }
    auto I_syn = target_state.get("I_syn");

    // Compute effective connectivity: weights * structural_mask * modulatory_mask.
    auto W_eff = connectivity_.effective_weights();

    const auto& bi = connectivity_.block_index;
    int64_t B = connectivity_.block_size;
    int64_t n_src = connectivity_.n_source;
    int64_t n_tgt = connectivity_.n_target;
    int64_t n_tgt_blocks = bi.n_rows();

    // Access CSR structure (on CPU for block iteration).
    auto row_ptr_cpu = bi.row_ptr.to(torch::kCPU);
    auto col_idx_cpu = bi.col_idx.to(torch::kCPU);
    auto rp = row_ptr_cpu.accessor<int32_t, 1>();
    auto ci = col_idx_cpu.accessor<int32_t, 1>();

    // Iterate over target block rows using CSR structure.
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

            // Slice the block weight matrix to actual sizes.
            auto W_block = W_eff[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);

            // Source spikes for this block.
            auto spike_block = spikes.slice(0, s_begin, s_end);

            // Block matmul: [t_size, s_size] @ [s_size] -> [t_size]
            auto I_block = torch::mv(W_block, spike_block);

            // Accumulate into target I_syn.
            I_syn.slice(0, t_begin, t_end) += I_block;
        }
    }
}

void Connection::attach(std::shared_ptr<PlasticityRule> rule) {
    rule->initialize(connectivity_);
    plasticity_rules_.push_back(std::move(rule));
}

void Connection::update_plasticity(const State& pre_state,
                                   const State& post_state,
                                   Time t, Duration dt) {
    for (auto& rule : plasticity_rules_) {
        rule->update(connectivity_, pre_state, post_state, t, dt);
    }
}

// ---------------------------------------------------------------------------
// Free function: connect()
// ---------------------------------------------------------------------------

std::shared_ptr<Connection> connect(
    std::shared_ptr<Population> source,
    std::shared_ptr<Population> target,
    TopologyGenerator& topology,
    std::shared_ptr<torch::nn::Module> synapse,
    const ConnectOptions& options) {
    // Use the topology generator to create the block structure and masks.
    // For now we pass CPU as default device; the caller can .to(device) later.
    auto ct = topology.generate(
        source->size(),
        target->size(),
        options.block_size(),
        torch::kCPU);

    return std::make_shared<Connection>(
        std::move(source),
        std::move(target),
        std::move(ct),
        std::move(synapse),
        options);
}

} // namespace nrn
