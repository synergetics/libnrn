#pragma once

#include <torch/torch.h>

#include <nrn/core/options.h>
#include <nrn/core/types.h>

namespace nrn {

/// Builder-pattern options for configuring a Simulation.
struct SimulationOptions {
    NRN_ARG(Duration, dt_fast,          0.0001);   // 0.1 ms
    NRN_ARG(Duration, dt_slow,          0.01);     // 10  ms
    NRN_ARG(Duration, dt_structural,    1.0);      // 1   s
    NRN_ARG(Duration, duration,         1.0);      // 1   s  total
    NRN_ARG(torch::Device, device,      torch::kCUDA);
    NRN_ARG(bool,     use_cuda_graphs,  false);
    NRN_ARG(int64_t,  max_delay_steps,  50);
};

} // namespace nrn
