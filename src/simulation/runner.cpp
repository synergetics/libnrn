#include <nrn/simulation/runner.h>

#include <cmath>

#include <nrn/core/module.h>
#include <nrn/connectivity/connect.h>
#include <nrn/graph/graph.h>
#include <nrn/graph/graph_runner.h>
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

    // Pre-create spike buffers for every population on the target device.
    auto device = sim->options.device();
    if (region) {
        for (const auto& pop : region->populations) {
            sim->spike_buffers.emplace(
                pop->name,
                SpikeBuffer(pop->n,
                            sim->options.max_delay_steps(),
                            device));
        }

        // Initialize population states by calling forward(dt=0).
        // This populates state bags on CPU first.
        for (auto& pop : region->populations) {
            nrn_forward(&pop->module, pop->state, 0.0, 0.0);
        }

        // Migrate all populations and connections to target device.
        if (device != torch::kCPU) {
            for (auto& pop : region->populations) {
                population_to_device(pop.get(), device);
            }
            for (auto& conn : region->connections) {
                conn->connectivity.to(device);
            }
            // Re-call forward(dt=0) to restore state bag aliasing.
            // After to_device(), the state bag tensors and the neuron struct
            // tensors are different GPU tensors. Re-publishing via forward()
            // makes the state bag point at the neuron struct's GPU tensors.
            for (auto& pop : region->populations) {
                nrn_forward(&pop->module, pop->state, 0.0, 0.0);
            }
        }

        // Compile the region into an execution graph.
        sim->graph = graph_compile(region, device);
    }

    return sim;
}

void sim_destroy(Simulation* sim) {
    if (sim->graph) {
        graph_destroy(sim->graph);
        sim->graph = nullptr;
    }
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
    if (sim->graph) {
        graph_step(sim->graph,
                   sim->clock,
                   sim->options.dt_fast(),
                   sim->spike_buffers,
                   sim->recorders,
                   sim->callbacks,
                   sim);
    }
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
