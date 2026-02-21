#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>

namespace nrn {

/// Non-template abstract interface for modules that can be stepped in time.
///
/// This enables Simulation::step() to call forward() through a
/// torch::nn::Module pointer via dynamic_cast<SteppableModule*>.
class SteppableModule {
public:
    virtual ~SteppableModule() = default;
    virtual void forward(State& state, Time t, Duration dt) = 0;
    virtual std::vector<std::string> state_vars() const = 0;
    virtual int64_t size() const = 0;
};

/// CRTP base module for all simulation components.
///
/// Inherits from torch::nn::Cloneable<Derived> to get parameter registration,
/// serialization, device transfer, and the TORCH_MODULE wrapper pattern.
/// Also inherits SteppableModule for type-erased virtual dispatch.
///
/// Derived classes must implement:
///   - reset()                          — initialize/reinitialize parameters
///   - forward(State&, Time, Duration)  — advance state by one timestep
///   - state_vars()                     — list the names of state variables
template <typename Derived>
class Module : public torch::nn::Cloneable<Derived>,
               public SteppableModule {
public:
    using torch::nn::Cloneable<Derived>::Cloneable;

    /// Advance the module state by one timestep.
    /// Subclasses override this to implement their dynamics.
    void forward(State& state, Time t, Duration dt) override {
        TORCH_CHECK(false,
                    "forward() not implemented for module '",
                    this->name(), "'");
    }

    /// Return the list of state variable names this module manages.
    std::vector<std::string> state_vars() const override {
        return {};
    }

    /// Return the number of cells / units in this module.
    int64_t size() const override { return n_; }

protected:
    int64_t n_ = 0; ///< Number of neurons / cells / units.
};

} // namespace nrn
