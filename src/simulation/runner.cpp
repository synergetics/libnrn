#include <nrn/simulation/runner.h>

#include <cmath>
#include <stdexcept>

#include <nrn/monitor/spike_recorder.h>
#include <nrn/monitor/state_recorder.h>

namespace nrn {

Simulation::Simulation(Region region, SimulationOptions options)
    : region_(std::move(region)),
      options_(std::move(options)),
      clock_(options_.dt_fast(),
             options_.dt_slow(),
             options_.dt_structural()) {
    // Pre-create spike buffers for every population in the region.
    if (region_) {
        for (const auto& pop : region_->populations()) {
            spike_buffers_.emplace(
                pop->name(),
                SpikeBuffer(pop->size(),
                            options_.max_delay_steps(),
                            options_.device()));
        }
    }
}

// ------------------------------------------------------------------
// Recording
// ------------------------------------------------------------------

void Simulation::record(std::shared_ptr<Population> population,
                        std::vector<std::string> variables,
                        int64_t subsample) {
    TORCH_CHECK(population != nullptr, "Cannot record a null population");

    // Check if "spike" is among the requested variables.
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
        recorders_.push_back(std::make_shared<SpikeRecorder>(
            population->name(), subsample));
    }
    if (!state_vars.empty()) {
        recorders_.push_back(std::make_shared<StateRecorder>(
            population->name(), std::move(state_vars), subsample));
    }
}

// ------------------------------------------------------------------
// Running
// ------------------------------------------------------------------

void Simulation::run() {
    int64_t total_steps = static_cast<int64_t>(
        std::round(options_.duration() / options_.dt_fast()));
    run_steps(total_steps);
}

void Simulation::run_steps(int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        step();
    }
}

void Simulation::step() {
    TORCH_CHECK(false,
                "Simulation::step() not yet implemented — the full "
                "time-stepping pipeline (synaptic delivery, neuron update, "
                "plasticity, slow subsystems) is planned for Phase 1");
}

// ------------------------------------------------------------------
// Data access
// ------------------------------------------------------------------

torch::Tensor Simulation::get_spikes(
    const std::shared_ptr<Population>& population) const {
    TORCH_CHECK(population != nullptr, "Cannot query spikes for null population");
    for (const auto& rec : recorders_) {
        if (rec->population_name() == population->name()) {
            auto* sr = dynamic_cast<SpikeRecorder*>(rec.get());
            if (sr) {
                return sr->get_spikes();
            }
        }
    }
    TORCH_CHECK(false, "No spike recorder registered for population '",
                population->name(), "'");
    return {}; // unreachable
}

torch::Tensor Simulation::get_recorded(
    const std::shared_ptr<Population>& population,
    const std::string& variable) const {
    TORCH_CHECK(population != nullptr,
                "Cannot query recorded data for null population");
    for (const auto& rec : recorders_) {
        if (rec->population_name() == population->name()) {
            auto* sr = dynamic_cast<StateRecorder*>(rec.get());
            if (sr) {
                return sr->get(variable);
            }
        }
    }
    TORCH_CHECK(false, "No state recorder registered for population '",
                population->name(), "' variable '", variable, "'");
    return {}; // unreachable
}

// ------------------------------------------------------------------
// I/O
// ------------------------------------------------------------------

void Simulation::save(const std::string& /*path*/) const {
    TORCH_CHECK(false,
                "Simulation::save() not yet implemented — HDF5 output is "
                "planned for Phase 1");
}

// ------------------------------------------------------------------
// Callbacks
// ------------------------------------------------------------------

void Simulation::add_callback(int64_t every_n, Callback fn) {
    TORCH_CHECK(every_n > 0, "Callback interval must be positive");
    callbacks_.emplace_back(every_n, std::move(fn));
}

} // namespace nrn
