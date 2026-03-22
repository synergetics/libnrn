/// @file graph_runner.h
/// @brief Single-step advance of the compiled execution Graph.
///
/// @details
/// Exposes graph_step(), which implements the 7-phase simulation loop
/// operating on NodeGroups and EdgeGroups rather than the higher-level
/// Population / Connection / Region objects.
///
/// @see Graph, graph_compile(), Simulation

#pragma once

#include <nrn/graph/graph.h>
#include <nrn/simulation/clock.h>
#include <nrn/simulation/spike_buffer.h>
#include <nrn/monitor/recorder.h>

#include <functional>
#include <unordered_map>
#include <vector>

namespace nrn {

// Forward declaration.
struct Simulation;

/// @brief Advance the simulation by one fast timestep using the compiled Graph.
///
/// @details
/// Executes the 7-phase loop:
///   1. **Zero I_syn** — clears the synaptic input current for every NodeGroup.
///   2. **Deliver synaptic input** — iterates EdgeGroups and calls
///      edge_group_deliver_spike() or edge_group_deliver_continuous().
///   3. **Forward-integrate neurons** — calls nrn_forward() on each NodeGroup's
///      NrnModule, updating @c V, @c spikes, adaptation variables, etc.
///   4. **Push spikes** — reads @c spikes from each NodeGroup's State and
///      records them in the corresponding SpikeBuffer for delayed delivery.
///   5. **Update plasticity** — calls edge_group_update_plasticity() for every
///      EdgeGroup that has plasticity rules attached.
///   6. **Record state** — invokes each active Recorder to sample the chosen
///      state variables.
///   7. **Advance clock / fire callbacks** — ticks the Clock and executes any
///      callbacks whose scheduled tick has been reached.
///
/// @param graph         Compiled execution graph; must not be null.
/// @param clock         Simulation clock tracking the current time and tick count.
/// @param dt            Fast-timescale timestep duration in seconds.
/// @param spike_buffers Map from population name to SpikeBuffer managing the
///                      delay lines for that population's spike output.
/// @param recorders     Active Recorder instances that sample state each step.
/// @param callbacks     Scheduled callbacks as (tick_id, fn) pairs; @p fn is
///                      called with the current Simulation and wall time when
///                      the tick matches.
/// @param sim           Non-owning pointer to the parent Simulation, forwarded
///                      to callbacks; must remain valid for the duration of the
///                      step.
///
/// @note This function is invoked once per fast-timescale tick by the
///       Simulation loop.  Multi-timescale operator splitting (slow /
///       structural / global ticks) is handled at the Simulation level.
///
/// @see Graph, graph_compile(), Clock, SpikeBuffer, Recorder
void graph_step(Graph* graph,
                Clock& clock,
                double dt,
                std::unordered_map<std::string, SpikeBuffer>& spike_buffers,
                std::vector<Recorder>& recorders,
                const std::vector<std::pair<int64_t, std::function<void(const Simulation&, double)>>>& callbacks,
                const Simulation* sim);

} // namespace nrn
