#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace nrn {

/// State is a named tensor bag — just a hashmap from name to tensor.
using State = std::unordered_map<std::string, torch::Tensor>;

// ---------------------------------------------------------------------------
// Free functions operating on State
// ---------------------------------------------------------------------------

/// Insert or overwrite a named tensor.
inline void state_set(State& s, const std::string& name, torch::Tensor t) {
    s[name] = std::move(t);
}

/// Retrieve a named tensor. Aborts if not found.
inline torch::Tensor& state_get(State& s, const std::string& name) {
    auto it = s.find(name);
    TORCH_CHECK(it != s.end(), "State does not contain tensor '", name, "'");
    return it->second;
}

inline const torch::Tensor& state_get(const State& s, const std::string& name) {
    auto it = s.find(name);
    TORCH_CHECK(it != s.end(), "State does not contain tensor '", name, "'");
    return it->second;
}

/// Check whether a named tensor exists.
inline bool state_contains(const State& s, const std::string& name) {
    return s.count(name) > 0;
}

/// Remove a named tensor. Returns true if it was present.
inline bool state_remove(State& s, const std::string& name) {
    return s.erase(name) > 0;
}

/// Return all tensor names.
inline std::vector<std::string> state_keys(const State& s) {
    std::vector<std::string> k;
    k.reserve(s.size());
    for (const auto& [name, _] : s) {
        k.push_back(name);
    }
    return k;
}

/// Move all tensors to the given device.
inline void state_to_device(State& s, torch::Device device) {
    for (auto& [name, tensor] : s) {
        tensor = tensor.to(device);
    }
}

/// Factory: create a State with zero-initialized tensors for the given
/// variable names, each of shape [n].
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
