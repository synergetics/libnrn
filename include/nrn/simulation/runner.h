#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/monitor/recorder.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/clock.h>
#include <nrn/simulation/options.h>
#include <nrn/simulation/spike_buffer.h>

namespace nrn {

/// Top-level simulation driver.
///
/// Owns the simulation clock, spike buffers, recorders, and orchestrates
/// the multi-rate time-stepping loop over a Region.
class Simulation {
public:
    /// User-supplied callback type: invoked every `n` fast steps.
    using Callback = std::function<void(const Simulation&, Time)>;

    /// Construct a simulation from a region and options.
    Simulation(Region region, SimulationOptions options = {});

    // ------------------------------------------------------------------
    // Recording
    // ------------------------------------------------------------------

    /// Record selected state variables from a population.
    ///
    /// @param population  The population to record from.
    /// @param variables   Names of state variables to record (e.g. {"v", "spike"}).
    /// @param subsample   Record every `subsample`-th fast timestep.
    void record(std::shared_ptr<Population> population,
                std::vector<std::string> variables,
                int64_t subsample = 1);

    // ------------------------------------------------------------------
    // Running
    // ------------------------------------------------------------------

    /// Run the full simulation for the configured duration.
    void run();

    /// Run for exactly `n` fast timesteps.
    void run_steps(int64_t n);

    /// Advance by a single fast timestep.
    void step();

    // ------------------------------------------------------------------
    // Data access
    // ------------------------------------------------------------------

    /// Retrieve spike data for a population.
    ///
    /// @return Tensor of shape [K, 2] — columns are (neuron_id, time).
    torch::Tensor get_spikes(const std::shared_ptr<Population>& population) const;

    /// Retrieve recorded state variable history.
    ///
    /// @return Tensor of shape [T, N].
    torch::Tensor get_recorded(const std::shared_ptr<Population>& population,
                               const std::string& variable) const;

    // ------------------------------------------------------------------
    // I/O
    // ------------------------------------------------------------------

    /// Save all recorded data to an HDF5 (or stub) file.
    void save(const std::string& path) const;

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Access the simulation clock.
    const Clock& clock() const { return clock_; }

    // ------------------------------------------------------------------
    // Callbacks
    // ------------------------------------------------------------------

    /// Register a callback to be invoked every `every_n` fast steps.
    void add_callback(int64_t every_n, Callback fn);

private:
    Region region_;
    SimulationOptions options_;
    Clock clock_;

    /// Per-population spike buffers (keyed by population name).
    std::unordered_map<std::string, SpikeBuffer> spike_buffers_;

    /// All active recorders.
    std::vector<std::shared_ptr<Recorder>> recorders_;

    /// (interval, callback) pairs.
    std::vector<std::pair<int64_t, Callback>> callbacks_;
};

} // namespace nrn
