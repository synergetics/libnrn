#pragma once

#include <optional>
#include <string>

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/core/state.h>
#include <nrn/core/types.h>

namespace nrn {

/// A named group of neurons (or other cells) sharing the same model.
///
/// Plain struct — holds a type-erased NrnModule handle, the runtime State,
/// a name, and optional spatial positions.
struct Population {
    std::string name;
    NrnModule module;       // ops-based handle to the neuron model
    State state;
    int64_t n;
    std::optional<torch::Tensor> positions;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Create a population. The NrnModule should already be initialized
/// (e.g. via lif_as_module(lif_create(...))).
Population* population_create(const std::string& name,
                              NrnModule module,
                              int64_t n,
                              torch::Device device);

/// Destroy a population. Does NOT destroy the underlying module —
/// that is the caller's responsibility.
void population_destroy(Population* pop);

/// Assign 3-D positions (shape [N, 3]) for distance-dependent connectivity.
void population_set_positions(Population* pop, torch::Tensor positions);

/// Move all state tensors and positions to the given device.
void population_to_device(Population* pop, torch::Device device);

} // namespace nrn
