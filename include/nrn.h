#pragma once

/// Umbrella header for the libnrn framework.
///
/// Including this single header gives access to all public API types.

// Core
#include <nrn/core/types.h>
#include <nrn/core/options.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>

// Neuron models
#include <nrn/neuron/neuron.h>

// Synapse models
#include <nrn/synapse/synapse.h>

// Connectivity
#include <nrn/connectivity/options.h>
#include <nrn/connectivity/connectivity_tensor.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/topology.h>
#include <nrn/connectivity/topology/random.h>

// Plasticity
#include <nrn/plasticity/plasticity_rule.h>
#include <nrn/plasticity/options.h>
#include <nrn/plasticity/stdp.h>

// Network composition
#include <nrn/network/network.h>

// Simulation
#include <nrn/simulation/options.h>
#include <nrn/simulation/clock.h>
#include <nrn/simulation/spike_buffer.h>
#include <nrn/simulation/runner.h>

// Monitoring
#include <nrn/monitor/recorder.h>
