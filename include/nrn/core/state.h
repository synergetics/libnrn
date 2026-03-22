/// @file state.h
/// @brief Named tensor bag (State) and its free-function API.
///
/// @details
/// @c State is the runtime communication bus between neuron models, synapse
/// delivery, plasticity, and recording subsystems.  It is a plain
/// @c std::unordered_map<std::string, torch::Tensor> aliased as @c nrn::State.
///
/// Each neuron model publishes a fixed set of named tensors (e.g. @c "v",
/// @c "spikes", @c "I_syn") into its population's State on the first
/// @c nrn_forward() call.  All subsequent graph phases read and write these
/// tensors by name.
///
/// All tensors are SoA layout: shape @c [N] where @c N is the number of
/// neurons in the owning NodeGroup / Population.
///
/// @see NodeGroup, Population, nrn_forward()

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace nrn {

/// @brief Named tensor bag — the runtime state of a neuron group.
///
/// @details
/// A @c std::unordered_map from variable name to a 1-D @c torch::Tensor of
/// shape @c [N].  Neuron models publish their state variables (e.g. @c "v",
/// @c "spikes") into this map; edge delivery and recorders consume them.
using State = std::unordered_map<std::string, torch::Tensor>;

// ---------------------------------------------------------------------------
// Free functions operating on State
// ---------------------------------------------------------------------------

/// @brief Insert or overwrite a named tensor.
///
/// @param s     State map to modify.
/// @param name  Key for the tensor.
/// @param t     Tensor to store (moved into the map).
inline void state_set(State& s, const std::string& name, torch::Tensor t) {
    s[name] = std::move(t);
}

/// @brief Retrieve a named tensor by mutable reference.
///
/// @param s     State map to query.
/// @param name  Key to look up.
/// @return Reference to the stored tensor.
///
/// @note Aborts via @c TORCH_CHECK if the key is not present.
inline torch::Tensor& state_get(State& s, const std::string& name) {
    auto it = s.find(name);
    TORCH_CHECK(it != s.end(), "State does not contain tensor '", name, "'");
    return it->second;
}

/// @brief Retrieve a named tensor by const reference.
///
/// @param s     State map to query.
/// @param name  Key to look up.
/// @return Const reference to the stored tensor.
///
/// @note Aborts via @c TORCH_CHECK if the key is not present.
inline const torch::Tensor& state_get(const State& s, const std::string& name) {
    auto it = s.find(name);
    TORCH_CHECK(it != s.end(), "State does not contain tensor '", name, "'");
    return it->second;
}

/// @brief Check whether a named tensor exists in the state.
///
/// @param s     State map to query.
/// @param name  Key to check.
/// @return @c true if the key is present.
inline bool state_contains(const State& s, const std::string& name) {
    return s.count(name) > 0;
}

/// @brief Remove a named tensor from the state.
///
/// @param s     State map to modify.
/// @param name  Key to remove.
/// @return @c true if the key was present and removed.
inline bool state_remove(State& s, const std::string& name) {
    return s.erase(name) > 0;
}

/// @brief Return all tensor names currently in the state.
///
/// @param s  State map to query.
/// @return Vector of key strings (order unspecified).
inline std::vector<std::string> state_keys(const State& s) {
    std::vector<std::string> k;
    k.reserve(s.size());
    for (const auto& [name, _] : s) {
        k.push_back(name);
    }
    return k;
}

/// @brief Move all tensors in the state to the given device in place.
///
/// @param s       State map to modify.
/// @param device  Target device.
inline void state_to_device(State& s, torch::Device device) {
    for (auto& [name, tensor] : s) {
        tensor = tensor.to(device);
    }
}

/// @brief Factory: create a State with zero-initialized tensors.
///
/// @details
/// Creates one zero tensor of shape @c [n] for each name in @p var_names.
///
/// @param var_names  Variable names to initialise.
/// @param n          Number of elements per tensor (number of neurons).
/// @param device     Device for tensor allocation (default: CPU).
/// @param dtype      Data type (default: @c float32).
/// @return Fully populated State with all tensors zeroed.
inline State state_create(const std::vector<std::string>& var_names,
                          int64_t n,
                          torch::Device device = torch::kCPU,
                          torch::Dtype dtype = torch::kFloat32) {
    State s;
    auto opts = torch::TensorOptions().device(device).dtype(dtype);
    for (const auto& name : var_names) {
        s[name] = torch::zeros({n}, opts);
    }
    return s;
}

} // namespace nrn
