/// @file population.h
/// @brief Population struct — a named group of neurons sharing one NrnModule.
///
/// @details
/// Population is a plain C++ struct (no virtual methods) that bundles:
///   - a type-erased @c NrnModule handle to the neuron model,
///   - a @c State named-tensor bag (populated on first @c nrn_forward() call),
///   - the neuron count @c n, and
///   - optional 3-D spatial positions.
///
/// Populations are owned by a @c Region via @c shared_ptr and are referenced
/// by the compiled Graph via non-owning @c State* pointers — **the Region must
/// outlive the Graph**.
///
/// @see Region, NrnModule, State, NodeGroup

#pragma once

#include <optional>
#include <string>

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/core/state.h>
#include <nrn/core/types.h>

namespace nrn {

/// @brief A named group of neurons sharing a single neuron model.
///
/// @details
/// Plain struct; all polymorphism is expressed through @c NrnModule::ops.
/// The @c state field is empty at construction; it is populated by the
/// first @c nrn_forward() call when the model publishes its state variables.
struct Population {
    std::string name;    ///< Human-readable population identifier.
    NrnModule module;    ///< Type-erased handle to the neuron model.
    State state;         ///< Named tensor bag; populated on first forward pass.
    int64_t n;           ///< Number of neurons.
    std::optional<torch::Tensor> positions; ///< Optional spatial positions @c [n, 3] (metres).
};

/// @brief Allocate and initialise a Population.
///
/// @details
/// The @p module handle must already be fully initialised
/// (e.g. via @c lif_as_module(lif_create(…))).
///
/// @param name    Human-readable name.
/// @param module  Type-erased neuron model handle.
/// @param n       Number of neurons (must be > 0).
/// @param device  Device hint (currently unused; tensors are allocated on device when model runs).
/// @return Heap-allocated Population; caller takes ownership.
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

/// @brief Free a Population.
/// @param pop  Pointer to the Population to destroy.
inline void population_destroy(Population* pop) {
    delete pop;
}

/// @brief Set the spatial positions of neurons in the population.
///
/// @param pop        Population to modify.
/// @param positions  Tensor of shape @c [n, 3] giving @c (x, y, z) in metres.
inline void population_set_positions(Population* pop, torch::Tensor positions) {
    TORCH_CHECK(positions.dim() == 2 && positions.size(0) == pop->n &&
                    positions.size(1) == 3,
                "positions must have shape [", pop->n, ", 3], got ",
                positions.sizes());
    pop->positions = std::move(positions);
}

/// @brief Move all tensors in the population (state + positions + module) to @p device.
/// @param pop     Population to modify.
/// @param device  Target device.
inline void population_to_device(Population* pop, torch::Device device) {
    state_to_device(pop->state, device);
    if (pop->positions.has_value()) {
        pop->positions = pop->positions->to(device);
    }
    nrn_to_device(&pop->module, device);
}

/// @brief Return a human-readable string representation for debugging.
/// @param pop  Population to describe.
/// @return Descriptive string.
inline std::string population_repr(const Population* pop) {
    std::string repr = "Population(name='" + pop->name + "', n=" + std::to_string(pop->n) + ", module=...)";
    if (pop->positions.has_value()) {
        repr += " with positions";
    }
    return repr;
}

} // namespace nrn
