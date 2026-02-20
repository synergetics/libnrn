#pragma once

/// @file nrn.h
/// Master include for the libnrn library.
///
/// Including this single header pulls in every public component:
/// core types, neuron models, synapse models, connectivity, plasticity,
/// network composition, simulation runner, monitoring, and configuration.
/// It also brings the unit-literal operators into scope.

// -- Core ------------------------------------------------------------------
#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/core/options.h>
#include <nrn/core/device.h>

// -- Neuron models ---------------------------------------------------------
#include <nrn/neuron/neuron.h>

// -- Synapse models --------------------------------------------------------
#include <nrn/synapse/synapse.h>

// -- Connectivity ----------------------------------------------------------
#include <nrn/connectivity/connectivity.h>

// -- Plasticity ------------------------------------------------------------
#include <nrn/plasticity/plasticity.h>

// -- Network composition ---------------------------------------------------
#include <nrn/network/network.h>

// -- Simulation ------------------------------------------------------------
#include <nrn/simulation/simulation.h>

// -- Monitoring / recording ------------------------------------------------
#include <nrn/monitor/monitor.h>

// -- Configuration / YAML --------------------------------------------------
#include <nrn/config/config.h>

// Bring unit-literal operators (e.g. 0.1_ms, -70.6_mV) into scope.
using namespace nrn::literals;
