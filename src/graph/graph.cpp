#include <nrn/graph/graph.h>

#include <algorithm>
#include <unordered_map>

#include <nrn/connectivity/connect.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>

namespace nrn {

/** Convert a block-sparse ConnectivityTensor to a dense matrix.
 * Used when Auto representation resolves to Dense. This is a one-time cost at graph compilation, and allows us to leverage efficient dense operations for small, dense circuits.
*/
static torch::Tensor materialize_dense(const ConnectivityTensor& ct) {
    int64_t n_tgt = ct.n_target;
    int64_t n_src = ct.n_source;
    int64_t B = ct.block_size;

    auto W_eff = ct.effective_weights();
    auto dense = torch::zeros({n_tgt, n_src}, W_eff.options());

    auto row_ptr_cpu = ct.block_index.row_ptr.to(torch::kCPU);
    auto col_idx_cpu = ct.block_index.col_idx.to(torch::kCPU);
    auto rp = row_ptr_cpu.accessor<int32_t, 1>();
    auto ci = col_idx_cpu.accessor<int32_t, 1>();

    int64_t n_tgt_blocks = ct.block_index.n_rows();

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
            dense.slice(0, t_begin, t_end).slice(1, s_begin, s_end).copy_(W_block);
        }
    }

    return dense;
}

// ---------------------------------------------------------------------------
// should_use_dense — heuristic for Auto representation selection
// ---------------------------------------------------------------------------

static bool should_use_dense(const ConnectivityTensor& ct) {
    int64_t n_entries = ct.n_source * ct.n_target;
    double density = ct.density();

    // Dense is preferred for small, dense circuits.
    // Threshold: density > 0.3 and total entries < 10M.
    return density > 0.3 && n_entries < 10'000'000;
}

// ---------------------------------------------------------------------------
// graph_compile
// ---------------------------------------------------------------------------

Graph* graph_compile(Region* region, torch::Device device) {
    TORCH_CHECK(region != nullptr, "graph_compile: region must not be null");

    auto* graph = new Graph();
    graph->device = device;

    // Build a map from Population pointer to node group index for edge lookup.
    std::unordered_map<Population*, uint32_t> pop_to_group;

    int64_t offset = 0;
    for (size_t i = 0; i < region->populations.size(); ++i) {
        auto& pop = region->populations[i];

        NodeGroup ng;
        ng.id = static_cast<uint32_t>(i);
        ng.name = pop->name;
        ng.module = pop->module;
        ng.state = &pop->state;
        ng.offset = offset;
        ng.count = pop->n;
        ng.neuron_group_id = -1;  // point neurons
        ng.is_soma = true;        // all point neurons emit spikes

        pop_to_group[pop.get()] = ng.id;
        offset += pop->n;

        graph->node_groups.push_back(std::move(ng));
    }
    graph->total_nodes = offset;

    // Create edge groups from connections.
    for (size_t i = 0; i < region->connections.size(); ++i) {
        auto& conn = region->connections[i];

        auto src_it = pop_to_group.find(conn->source.get());
        auto tgt_it = pop_to_group.find(conn->target.get());

        TORCH_CHECK(src_it != pop_to_group.end(),
                    "graph_compile: source population '", conn->source->name,
                    "' not found in region");
        TORCH_CHECK(tgt_it != pop_to_group.end(),
                    "graph_compile: target population '", conn->target->name,
                    "' not found in region");

        EdgeGroup eg;
        eg.id = static_cast<uint32_t>(i);
        eg.source_group_id = src_it->second;
        eg.target_group_id = tgt_it->second;
        eg.trigger = conn->options.trigger();
        eg.synapse = conn->synapse;
        eg.plasticity_rules = conn->plasticity_rules;

        // Resolve representation.
        EdgeRepresentation repr = EdgeRepresentation::BlockSparse;
        if (should_use_dense(conn->connectivity)) {
            repr = EdgeRepresentation::Dense;
        }
        eg.representation = repr;

        if (repr == EdgeRepresentation::Dense) {
            // Materialize block-sparse into dense matrix.
            eg.dense_weights = materialize_dense(conn->connectivity);
            eg.block_sparse = nullptr;

            // Initialize plasticity traces for dense representation.
            if (!conn->plasticity_rules.empty()) {
                eg.dense_trace_pre = torch::zeros_like(eg.dense_weights);
                eg.dense_trace_post = torch::zeros_like(eg.dense_weights);
            }
        } else {
            // Reference the existing block-sparse connectivity (non-owning).
            eg.block_sparse = &conn->connectivity;
        }

        graph->edge_groups.push_back(std::move(eg));
    }

    return graph;
}

// ---------------------------------------------------------------------------
// graph_destroy
// ---------------------------------------------------------------------------

void graph_destroy(Graph* graph) {
    delete graph;
}

// ---------------------------------------------------------------------------
// graph_find_node_group
// ---------------------------------------------------------------------------

NodeGroup* graph_find_node_group(Graph* graph, const std::string& name) {
    for (auto& ng : graph->node_groups) {
        if (ng.name == name) {
            return &ng;
        }
    }
    return nullptr;
}

} // namespace nrn
