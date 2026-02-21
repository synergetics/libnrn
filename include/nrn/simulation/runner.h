#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/monitor/recorder.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/clock.h>
#include <nrn/simulation/options.h>
#include <nrn/simulation/spike_buffer.h>

namespace nrn {

/// Forward declare for the callback signature.
struct Simulation;

/// User-supplied callback type.
using SimCallback = std::function<void(const Simulation&, double)>;

/// Top-level simulation driver.
struct Simulation {
    Region* region;
    SimulationOptions options;
    Clock clock;

    /// Per-population spike buffers (keyed by population name).
    std::unordered_map<std::string, SpikeBuffer> spike_buffers;

    /// All active recorders.
    std::vector<Recorder> recorders;

    /// (interval, callback) pairs.
    std::vector<std::pair<int64_t, SimCallback>> callbacks;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Create a simulation from a region and options.
Simulation* sim_create(Region* region, SimulationOptions options = {});

/// Destroy a simulation.
void sim_destroy(Simulation* sim);

/// Record selected state variables from a population.
void sim_record(Simulation* sim,
                std::shared_ptr<Population> population,
                std::vector<std::string> variables,
                int64_t subsample = 1);

/// Run the full simulation for the configured duration.
void sim_run(Simulation* sim);

/// Run for exactly n fast timesteps.
void sim_run_steps(Simulation* sim, int64_t n);

/// Advance by a single fast timestep.
void sim_step(Simulation* sim);

/// Retrieve spike data for a population.
/// Returns Tensor of shape [K, 2] — columns are (neuron_id, time).
torch::Tensor sim_get_spikes(const Simulation* sim,
                             const std::shared_ptr<Population>& population);

/// Retrieve recorded state variable history.
/// Returns Tensor of shape [T, N].
torch::Tensor sim_get_recorded(const Simulation* sim,
                               const std::shared_ptr<Population>& population,
                               const std::string& variable);

/// Register a callback to be invoked every every_n fast steps.
void sim_add_callback(Simulation* sim, int64_t every_n, SimCallback fn);

} // namespace nrn
