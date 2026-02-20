#pragma once

#include <optional>
#include <string>

#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>

namespace nrn {

/// A named group of neurons (or other cells) sharing the same model.
///
/// Population is *not* a Module — it is a thin wrapper that binds a
/// type-erased neuron module (stored as `torch::nn::AnyModule`) together
/// with its runtime `State`, a population name, and optional spatial
/// positions.  It is the primary unit that Region aggregates.
class Population {
public:
    /// Construct a population.
    ///
    /// @param name    Human-readable identifier (e.g. "excitatory").
    /// @param module  Neuron module (e.g. `nrn::neuron::LIF`).
    /// @param n       Number of neurons in this population.
    /// @param device  Torch device for state tensors.
    Population(std::string name,
               std::shared_ptr<torch::nn::Module> module,
               int64_t n,
               torch::Device device);

    /// Access the population's state bag (mutable).
    State& state() { return state_; }
    const State& state() const { return state_; }

    /// Access the wrapped neuron module.
    std::shared_ptr<torch::nn::Module>& module() { return module_; }
    const std::shared_ptr<torch::nn::Module>& module() const { return module_; }

    /// Population name.
    const std::string& name() const { return name_; }

    /// Number of neurons.
    int64_t size() const { return n_; }

    /// Assign 3-D positions (shape [N, 3]) for distance-dependent connectivity.
    void set_positions(torch::Tensor positions);

    /// Return positions if set, or nullopt.
    const std::optional<torch::Tensor>& positions() const { return positions_; }

    /// Move all tensors (state + positions) to the given device.
    void to(torch::Device device);

private:
    std::string name_;
    std::shared_ptr<torch::nn::Module> module_;
    State state_;
    int64_t n_;
    std::optional<torch::Tensor> positions_;
};

} // namespace nrn
