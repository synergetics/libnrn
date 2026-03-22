/// @file runner.h
/// @brief Simulation struct and its free-function driver API.
///
/// @details
/// @c Simulation is the top-level driver.  It owns the Clock, SpikeBuffers,
/// Recorders, and the compiled Graph.  The public API is a set of free
/// functions: @c sim_create(), @c sim_run(), @c sim_step(), @c sim_destroy().
///
/// @see sim_create(), sim_run(), sim_step(), Graph, graph_compile()

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

// Forward declarations.
struct Simulation;
struct Graph;

/// @brief User-supplied callback invoked at a fixed interval during simulation.
///
/// @details
/// Signature: @c void(const Simulation&, double t)
using SimCallback = std::function<void(const Simulation&, double)>;

/// @brief Top-level simulation driver.
///
/// @details
/// Owns:
///   - A non-owning pointer to the source @c Region (Region must outlive Simulation).
///   - A @c Clock tracking the multi-rate simulation time.
///   - Per-population @c SpikeBuffer delay lines.
///   - A list of active @c Recorder objects.
///   - A list of scheduled callbacks.
///   - The compiled @c Graph (owned, freed by @c sim_destroy()).
///
/// @see sim_create(), sim_run(), sim_step(), sim_destroy()
struct Simulation {
    Region* region;               ///< Non-owning pointer to source Region; must outlive Simulation.
    SimulationOptions options;    ///< Configuration (timesteps, device, etc.).
    Clock clock;                  ///< Multi-rate simulation clock.

    std::unordered_map<std::string, SpikeBuffer> spike_buffers; ///< Per-population delay-line spike buffers.
    std::vector<Recorder> recorders;                             ///< Active state / spike recorders.
    std::vector<std::pair<int64_t, SimCallback>> callbacks;      ///< (tick_interval, callback) pairs.

    Graph* graph = nullptr; ///< Compiled execution graph (owned; freed by sim_destroy()).
};

/// @brief Create and compile a Simulation from a Region.
///
/// @details
/// Calls @c graph_compile() internally.  The @p region must outlive
/// the returned Simulation.
///
/// @param region   Source Region; must not be null.
/// @param options  Simulation configuration.
/// @return Heap-allocated Simulation; caller must call @c sim_destroy().
Simulation* sim_create(Region* region, SimulationOptions options = {});

/// @brief Free a Simulation and its compiled Graph.
/// @param sim  Simulation to destroy; may be null (no-op).
void sim_destroy(Simulation* sim);

/// @brief Attach a state or spike recorder to a population.
///
/// @param sim        Simulation to modify.
/// @param population Population to observe.
/// @param variables  State variable names to record (e.g. @c {"v", "spike"}).
/// @param subsample  Record every @p subsample-th fast step (default: every step).
void sim_record(Simulation* sim,
                std::shared_ptr<Population> population,
                std::vector<std::string> variables,
                int64_t subsample = 1);

/// @brief Run the full simulation for SimulationOptions::duration.
/// @param sim  Simulation to run.
void sim_run(Simulation* sim);

/// @brief Run for exactly @p n fast timesteps.
/// @param sim  Simulation to advance.
/// @param n    Number of fast steps to execute.
void sim_run_steps(Simulation* sim, int64_t n);

/// @brief Advance by a single fast timestep.
/// @param sim  Simulation to advance.
void sim_step(Simulation* sim);

/// @brief Retrieve all recorded spike events for a population.
///
/// @param sim        Simulation to query.
/// @param population Population whose spikes to return.
/// @return Tensor of shape @c [K, 2] where column 0 is neuron index and column 1 is spike time.
torch::Tensor sim_get_spikes(const Simulation* sim,
                             const std::shared_ptr<Population>& population);

/// @brief Retrieve recorded history of a state variable.
///
/// @param sim        Simulation to query.
/// @param population Population to retrieve data for.
/// @param variable   Name of the state variable (e.g. @c "v").
/// @return Tensor of shape @c [T, N] (T samples, N neurons).
torch::Tensor sim_get_recorded(const Simulation* sim,
                               const std::shared_ptr<Population>& population,
                               const std::string& variable);

/// @brief Register a callback to be invoked every @p every_n fast steps.
///
/// @param sim      Simulation to modify.
/// @param every_n  Callback interval in fast timesteps.
/// @param fn       Callback function; receives the Simulation and current time.
void sim_add_callback(Simulation* sim, int64_t every_n, SimCallback fn);

} // namespace nrn
