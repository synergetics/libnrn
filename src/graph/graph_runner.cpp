#include <nrn/graph/graph_runner.h>
#include <nrn/graph/edge_ops.h>

#include <nrn/core/module.h>
#include <nrn/simulation/runner.h>

namespace nrn {

void graph_step(Graph* graph,
                Clock& clock,
                double dt,
                std::unordered_map<std::string, SpikeBuffer>& spike_buffers,
                std::vector<Recorder>& recorders,
                const std::vector<std::pair<int64_t, std::function<void(const Simulation&, double)>>>& callbacks,
                const Simulation* sim) {
    auto t = clock.time();

    // --- Phase 1: Zero I_syn for all node groups ---
    for (auto& ng : graph->node_groups) {
        if (state_contains(*ng.state, "I_syn")) {
            state_get(*ng.state, "I_syn").zero_();
        }
    }

    // --- Phase 2: Deliver through all edge groups ---
    for (auto& eg : graph->edge_groups) {
        auto& src = graph->node_groups[eg.source_group_id];
        auto& tgt = graph->node_groups[eg.target_group_id];

        if (eg.trigger == EdgeTrigger::Continuous) {
            edge_group_deliver_continuous(eg, src, tgt, t, dt);
        } else {
            // Spike-triggered: read from spike buffer.
            torch::Tensor source_spikes;
            auto it = spike_buffers.find(src.name);
            if (it != spike_buffers.end()) {
                source_spikes = it->second.read(1);
            } else if (state_contains(*src.state, "spike")) {
                source_spikes = state_get(*src.state, "spike");
            } else {
                continue;
            }

            edge_group_deliver_spike(eg, src, tgt, source_spikes, t, dt);
        }
    }

    // --- Phase 3: Forward-integrate all node groups ---
    for (auto& ng : graph->node_groups) {
        nrn_forward(&ng.module, *ng.state, t, dt);
    }

    // --- Phase 4: Push spikes into spike buffers ---
    for (auto& ng : graph->node_groups) {
        if (ng.is_soma && state_contains(*ng.state, "spike")) {
            auto it = spike_buffers.find(ng.name);
            if (it != spike_buffers.end()) {
                it->second.push(state_get(*ng.state, "spike"));
            }
        }
    }

    // --- Phase 5: Plasticity updates (at slow boundary) ---
    if (clock.is_slow_boundary()) {
        for (auto& eg : graph->edge_groups) {
            auto& src = graph->node_groups[eg.source_group_id];
            auto& tgt = graph->node_groups[eg.target_group_id];
            edge_group_update_plasticity(eg, *src.state, *tgt.state, t, dt);
        }
    }

    // --- Phase 6: Record state ---
    for (auto& rec : recorders) {
        for (auto& ng : graph->node_groups) {
            const char* rec_name = rec.ops->population_name(rec.impl);
            if (ng.name == rec_name) {
                rec.ops->record(rec.impl, *ng.state, t);
                break;
            }
        }
    }

    // --- Phase 7: Callbacks ---
    uint64_t step_num = clock.step();
    for (auto& [interval, callback] : callbacks) {
        if (step_num > 0 && (step_num % interval) == 0) {
            callback(*sim, t);
        }
    }

    // --- Advance clock ---
    clock.advance_fast();
}

} // namespace nrn
