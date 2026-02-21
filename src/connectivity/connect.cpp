#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/topology.h>
#include <nrn/network/population.h>
#include <nrn/plasticity/plasticity_rule.h>

#include <algorithm>

namespace nrn {

// ---------------------------------------------------------------------------
// Connection lifecycle
// ---------------------------------------------------------------------------

Connection* connection_create(std::shared_ptr<Population> source,
                              std::shared_ptr<Population> target,
                              ConnectivityTensor connectivity,
                              NrnModule synapse,
                              ConnectOptions options) {
    auto* conn = new Connection();
    conn->source = std::move(source);
    conn->target = std::move(target);
    conn->connectivity = std::move(connectivity);
    conn->synapse = synapse;
    conn->options = std::move(options);
    return conn;
}

void connection_destroy(Connection* conn) {
    delete conn;
}

// ---------------------------------------------------------------------------
// Spike delivery — block-sparse matmul through CSR
// ---------------------------------------------------------------------------

void connection_deliver(Connection* conn, const torch::Tensor& spikes,
                        double /*t*/, double /*dt*/) {
    auto& target_state = conn->target->state;
    if (!state_contains(target_state, "I_syn")) {
        return;  // Target state not yet initialized.
    }
    auto I_syn = state_get(target_state, "I_syn");

    auto W_eff = conn->connectivity.effective_weights();

    const auto& bi = conn->connectivity.block_index;
    int64_t B = conn->connectivity.block_size;
    int64_t n_src = conn->connectivity.n_source;
    int64_t n_tgt = conn->connectivity.n_target;
    int64_t n_tgt_blocks = bi.n_rows();

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
// Plasticity
// ---------------------------------------------------------------------------

void connection_attach(Connection* conn, PlasticityRule rule) {
    plasticity_initialize(&rule, conn->connectivity);
    conn->plasticity_rules.push_back(rule);
}

void connection_update_plasticity(Connection* conn,
                                  const State& pre_state,
                                  const State& post_state,
                                  double t, double dt) {
    for (auto& rule : conn->plasticity_rules) {
        plasticity_update(&rule, conn->connectivity, pre_state, post_state, t, dt);
    }
}

// ---------------------------------------------------------------------------
// Free function: connect()
// ---------------------------------------------------------------------------

std::shared_ptr<Connection> connect(
    std::shared_ptr<Population> source,
    std::shared_ptr<Population> target,
    TopologyGenerator* topology,
    NrnModule synapse,
    const ConnectOptions& options) {
    auto ct = topology_generate(topology,
                                source->n,
                                target->n,
                                options.block_size(),
                                torch::kCPU);

    auto* conn = connection_create(
        std::move(source),
        std::move(target),
        std::move(ct),
        synapse,
        options);

    return std::shared_ptr<Connection>(conn, connection_destroy);
}

} // namespace nrn
