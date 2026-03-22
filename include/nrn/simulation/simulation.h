/// @file simulation.h
/// @brief Aggregate include for the simulation subsystem.
///
/// @details
/// Pulls in Clock, SimulationOptions, the Simulation struct and its
/// free-function API, and SpikeBuffer.
///
/// @see Simulation, sim_create(), sim_run(), Clock, SpikeBuffer

#pragma once

#include <nrn/simulation/clock.h>
#include <nrn/simulation/options.h>
#include <nrn/simulation/runner.h>
#include <nrn/simulation/spike_buffer.h>
