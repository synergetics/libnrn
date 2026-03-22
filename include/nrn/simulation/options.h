/// @file options.h
/// @brief Builder-pattern options for configuring a Simulation.
///
/// @see Simulation, sim_create()

#pragma once

#include <torch/torch.h>

#include <nrn/core/options.h>
#include <nrn/core/types.h>

namespace nrn {

/// @brief Configuration options for a Simulation.
///
/// @details
/// Supports three timescales and optional CUDA Graph capture for fast
/// repeated execution of the inner simulation loop.
///
/// @see sim_create(), Clock
struct SimulationOptions {
    NRN_ARG(Duration, dt_fast,          0.0001);         ///< Fast (neural dynamics) timestep (s); default 0.1 ms.
    NRN_ARG(Duration, dt_slow,          0.01);           ///< Slow (modulatory) timestep (s); default 10 ms.
    NRN_ARG(Duration, dt_structural,    1.0);            ///< Structural plasticity timestep (s); default 1 s.
    NRN_ARG(Duration, duration,         1.0);            ///< Total simulation duration (s); default 1 s.
    NRN_ARG(torch::Device, device,      torch::kCUDA);   ///< Compute device; default CUDA.
    NRN_ARG(bool,     use_cuda_graphs,  false);          ///< Capture the inner loop as a CUDA Graph (reduces launch overhead).
    NRN_ARG(int64_t,  max_delay_steps,  50);             ///< Maximum synaptic delay in fast timesteps.
};

} // namespace nrn
