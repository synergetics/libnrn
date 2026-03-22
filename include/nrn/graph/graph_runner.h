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

/**
 * @brief Advance the simulation by one fast timestep using the Graph representation.
 *
 * Implements the same 7-phase loop as the original sim_step, but operates
 * on NodeGroups and EdgeGroups instead of Populations and Connections.
 *
 * @param graph Pointer to the neural network graph representation.
 * @param clock Reference to the simulation clock tracking current time.
 * @param dt The deterministic timestep duration for this advancement.
 * @param spike_buffers Map of spike buffers used to collect firing events.
 * @param recorders Collection of active recorders for capturing simulation states.
 * @param callbacks Vector of callback pairs containing a time identifier and a function to execute.
 * @param sim Pointer to the parent Simulation context.
 */
void graph_step(Graph* graph,
                Clock& clock,
                double dt,
                std::unordered_map<std::string, SpikeBuffer>& spike_buffers,
                std::vector<Recorder>& recorders,
                const std::vector<std::pair<int64_t, std::function<void(const Simulation&, double)>>>& callbacks,
                const Simulation* sim);

} // namespace nrn
