#include <nrn/simulation/runner.h>

#include <cmath>
#include <stdexcept>

#include <nrn/core/module.h>
#include <nrn/connectivity/connect.h>
#include <nrn/monitor/spike_recorder.h>
#include <nrn/monitor/state_recorder.h>

namespace nrn {

// ------------------------------------------------------------------
// Create / Destroy
// ------------------------------------------------------------------

Simulation* sim_create(Region* region, SimulationOptions options) {
    auto* sim = new Simulation();
    sim->region = region;
    sim->options = std::move(options);
    sim->clock = Clock(sim->options.dt_fast(),
                       sim->options.dt_slow(),
                       sim->options.dt_structural());

    // Pre-create spike buffers for every population.
    if (region) {
        for (const auto& pop : region->populations) {
            sim->spike_buffers.emplace(
                pop->name,
                SpikeBuffer(pop->n,
                            sim->options.max_delay_steps(),
                            torch::kCPU));
        }

        // Initialize population states by calling forward(dt=0).
        for (auto& pop : region->populations) {
            nrn_forward(&pop->module, pop->state, 0.0, 0.0);
        }
    }

    return sim;
}

void sim_destroy(Simulation* sim) {
    delete sim;
}

// ------------------------------------------------------------------
// Recording
// ------------------------------------------------------------------

void sim_record(Simulation* sim,
                std::shared_ptr<Population> population,
                std::vector<std::string> variables,
                int64_t subsample) {
    TORCH_CHECK(population != nullptr, "Cannot record a null population");

    bool wants_spikes = false;
    std::vector<std::string> state_vars;
    for (auto& v : variables) {
        if (v == "spike") {
            wants_spikes = true;
        } else {
            state_vars.push_back(std::move(v));
        }
    }

    if (wants_spikes) {
        auto* sr = spike_recorder_create(population->name, subsample);
        sim->recorders.push_back(spike_recorder_as_recorder(sr));
    }
    if (!state_vars.empty()) {
        auto* sr = state_recorder_create(population->name,
                                         std::move(state_vars), subsample);
        sim->recorders.push_back(state_recorder_as_recorder(sr));
    }
}

// ------------------------------------------------------------------
// Running
// ------------------------------------------------------------------

void sim_run(Simulation* sim) {
    int64_t total_steps = static_cast<int64_t>(
        std::round(sim->options.duration() / sim->options.dt_fast()));
    sim_run_steps(sim, total_steps);
}

void sim_run_steps(Simulation* sim, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        sim_step(sim);
    }
}

void sim_step(Simulation* sim) {
    auto t = sim->clock.time();
    auto dt = sim->options.dt_fast();

    // --- Phase 1: Zero I_syn for all populations ---
    for (auto& pop : sim->region->populations) {
        if (state_contains(pop->state, "I_syn")) {
            state_get(pop->state, "I_syn").zero_();
        }
    }

    // --- Phase 2: Deliver spikes through all connections ---
    for (auto& conn : sim->region->connections) {
        auto& source_name = conn->source->name;

        torch::Tensor source_spikes;
        auto it = sim->spike_buffers.find(source_name);
        if (it != sim->spike_buffers.end()) {
            source_spikes = it->second.read(1);
        } else {
            auto& src_state = conn->source->state;
            if (state_contains(src_state, "spike")) {
                source_spikes = state_get(src_state, "spike");
            } else {
                continue;
            }
        }

        connection_deliver(conn.get(), source_spikes, t, dt);
    }

    // --- Phase 3: Forward-integrate all neuron populations ---
    for (auto& pop : sim->region->populations) {
        nrn_forward(&pop->module, pop->state, t, dt);
    }

    // --- Phase 4: Push spikes into spike buffers ---
    for (auto& pop : sim->region->populations) {
        if (state_contains(pop->state, "spike")) {
            auto it = sim->spike_buffers.find(pop->name);
            if (it != sim->spike_buffers.end()) {
                it->second.push(state_get(pop->state, "spike"));
            }
        }
    }

    // --- Phase 5: Plasticity updates (at slow boundary) ---
    if (sim->clock.is_slow_boundary()) {
        for (auto& conn : sim->region->connections) {
            connection_update_plasticity(
                conn.get(),
                conn->source->state,
                conn->target->state,
                t, dt);
        }
    }

    // --- Phase 6: Record state ---
    for (auto& rec : sim->recorders) {
        for (auto& pop : sim->region->populations) {
            const char* rec_name = rec.ops->population_name(rec.impl);
            if (pop->name == rec_name) {
                rec.ops->record(rec.impl, pop->state, t);
                break;
            }
        }
    }

    // --- Phase 7: Callbacks ---
    uint64_t step_num = sim->clock.step();
    for (auto& [interval, callback] : sim->callbacks) {
        if (step_num > 0 && (step_num % interval) == 0) {
            callback(*sim, t);
        }
    }

    // --- Advance clock ---
    sim->clock.advance_fast();
}

// ------------------------------------------------------------------
// Data access
// ------------------------------------------------------------------

torch::Tensor sim_get_spikes(const Simulation* sim,
                             const std::shared_ptr<Population>& population) {
    TORCH_CHECK(population != nullptr,
                "Cannot query spikes for null population");
    for (const auto& rec : sim->recorders) {
        const char* rec_name = rec.ops->population_name(rec.impl);
        if (population->name == rec_name) {
            // Check if this is a spike recorder by trying to cast.
            // We stored the ops pointer, so we can compare it.
            if (rec.ops == &spike_recorder_ops) {
                auto* sr = static_cast<SpikeRecorderState*>(rec.impl);
                return spike_recorder_get_spikes(sr);
            }
        }
    }
    TORCH_CHECK(false, "No spike recorder registered for population '",
                population->name, "'");
    return {};
}

torch::Tensor sim_get_recorded(const Simulation* sim,
                               const std::shared_ptr<Population>& population,
                               const std::string& variable) {
    TORCH_CHECK(population != nullptr,
                "Cannot query recorded data for null population");
    for (const auto& rec : sim->recorders) {
        const char* rec_name = rec.ops->population_name(rec.impl);
        if (population->name == rec_name) {
            if (rec.ops == &state_recorder_ops) {
                auto* sr = static_cast<StateRecorderState*>(rec.impl);
                return state_recorder_get(sr, variable);
            }
        }
    }
    TORCH_CHECK(false, "No state recorder registered for population '",
                population->name, "' variable '", variable, "'");
    return {};
}

// ------------------------------------------------------------------
// I/O
// ------------------------------------------------------------------

// save() — stub, not yet implemented.

// ------------------------------------------------------------------
// Callbacks
// ------------------------------------------------------------------

void sim_add_callback(Simulation* sim, int64_t every_n, SimCallback fn) {
    TORCH_CHECK(every_n > 0, "Callback interval must be positive");
    sim->callbacks.emplace_back(every_n, std::move(fn));
}

} // namespace nrn
