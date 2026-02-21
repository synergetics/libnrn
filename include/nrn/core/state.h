#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace nrn {

/// Named tensor bag — holds named state tensors for a simulation component.
///
/// Provides dictionary-style access to tensors by name, used to pass and
/// inspect the complete state of neurons, synapses, or other modules.
class State {
public:
    State() = default;

    /// Factory: create a State with zero-initialized tensors for the given
    /// variable names, each of shape [n].
    static State create(const std::vector<std::string>& var_names,
                        int64_t n,
                        torch::Device device = torch::kCPU,
                        torch::Dtype dtype = torch::kFloat32) {
        State s;
        auto opts = torch::TensorOptions().device(device).dtype(dtype);
        for (const auto& name : var_names) {
            s.set(name, torch::zeros({n}, opts));
        }
        return s;
    }

    /// Insert or overwrite a named tensor.
    void set(const std::string& name, torch::Tensor tensor) {
        tensors_[name] = std::move(tensor);
    }

    /// Retrieve a named tensor. Throws if not found.
    torch::Tensor& get(const std::string& name) {
        auto it = tensors_.find(name);
        TORCH_CHECK(it != tensors_.end(),
                    "State does not contain tensor '", name, "'");
        return it->second;
    }

    const torch::Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        TORCH_CHECK(it != tensors_.end(),
                    "State does not contain tensor '", name, "'");
        return it->second;
    }

    /// Bracket operator — alias for get().
    torch::Tensor& operator[](const std::string& name) {
        return get(name);
    }

    const torch::Tensor& operator[](const std::string& name) const {
        return get(name);
    }

    /// Check whether a named tensor exists (alias: contains).
    bool has(const std::string& name) const {
        return tensors_.count(name) > 0;
    }

    /// Check whether a named tensor exists.
    bool contains(const std::string& name) const {
        return tensors_.count(name) > 0;
    }

    /// Remove a named tensor. Returns true if it was present.
    bool remove(const std::string& name) {
        return tensors_.erase(name) > 0;
    }

    /// Return all tensor names.
    std::vector<std::string> keys() const {
        std::vector<std::string> k;
        k.reserve(tensors_.size());
        for (const auto& [name, _] : tensors_) {
            k.push_back(name);
        }
        return k;
    }

    /// Number of tensors.
    size_t size() const { return tensors_.size(); }

    /// Move all tensors to the given device.
    void to(torch::Device device) {
        for (auto& [name, tensor] : tensors_) {
            tensor = tensor.to(device);
        }
    }

    /// Direct map access for iteration.
    auto begin() { return tensors_.begin(); }
    auto end() { return tensors_.end(); }
    auto begin() const { return tensors_.begin(); }
    auto end() const { return tensors_.end(); }

private:
    std::unordered_map<std::string, torch::Tensor> tensors_;
};

} // namespace nrn
