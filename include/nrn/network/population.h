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
inline Population* population_create(const std::string& name,
                              NrnModule module,
                              int64_t n,
                              torch::Device /*device*/) {
    TORCH_CHECK(n > 0, "Population '", name, "' must have at least 1 neuron");
    auto* pop = new Population();
    pop->name = name;
    pop->module = module;
    pop->n = n;
    pop->positions = std::nullopt;
    // State is empty at construction; populated by first forward() call.
    return pop;
}

// Destroy a population and free its resources.
inline void population_destroy(Population* pop) {
    delete pop;
}

// Set the positions of the neurons in the population.
// The positions tensor should have shape [n, 3].
inline void population_set_positions(Population* pop, torch::Tensor positions) {
    TORCH_CHECK(positions.dim() == 2 && positions.size(0) == pop->n &&
                    positions.size(1) == 3,
                "positions must have shape [", pop->n, ", 3], got ",
                positions.sizes());
    pop->positions = std::move(positions);
}

// Move the population's tensors to the specified device.
inline void population_to_device(Population* pop, torch::Device device) {
    state_to_device(pop->state, device);
    if (pop->positions.has_value()) {
        pop->positions = pop->positions->to(device);
    }
    nrn_to_device(&pop->module, device);
}

// Get a string representation of the population for debugging.
inline std::string population_repr(const Population* pop) {
    std::string repr = "Population(name='" + pop->name + "', n=" + std::to_string(pop->n) + ", module=...)";
    if (pop->positions.has_value()) {
        repr += " with positions";
    }
    return repr;
}

} // namespace nrn
