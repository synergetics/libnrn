#pragma once

#include <memory>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/connectivity/connectivity_tensor.h>
#include <nrn/connectivity/options.h>

namespace nrn {

// Forward declarations — avoid pulling in full headers.
class Population;
class PlasticityRule;
class TopologyGenerator;

// ---------------------------------------------------------------------------
// Connection — owns connectivity between a source and target population.
//
// Wraps a ConnectivityTensor and an optional synapse module, providing
// the interface for spike delivery, plasticity attachment, and inspection.
// ---------------------------------------------------------------------------
class Connection {
public:
    /// Construct a Connection from pre-built components.
    Connection(std::shared_ptr<Population> source,
               std::shared_ptr<Population> target,
               ConnectivityTensor connectivity,
               std::shared_ptr<torch::nn::Module> synapse,
               ConnectOptions options);

    /// Deliver presynaptic spikes through the connectivity to produce
    /// postsynaptic currents.  Stub — will integrate with spike buffer
    /// and CUDA kernel dispatch.
    void deliver(const torch::Tensor& spikes, Time t, Duration dt);

    /// Attach a plasticity rule that will operate on this connection's
    /// connectivity tensor.
    void attach(std::shared_ptr<PlasticityRule> rule);

    /// Run all attached plasticity rules for one update step.
    void update_plasticity(const State& pre_state,
                           const State& post_state,
                           Time t, Duration dt);

    /// Read-only access to the underlying connectivity tensor.
    const ConnectivityTensor& connectivity() const { return connectivity_; }

    /// Mutable access to the underlying connectivity tensor.
    ConnectivityTensor& connectivity() { return connectivity_; }

    /// Source population.
    std::shared_ptr<Population> source() const { return source_; }

    /// Target population.
    std::shared_ptr<Population> target() const { return target_; }

private:
    std::shared_ptr<Population> source_;
    std::shared_ptr<Population> target_;
    ConnectivityTensor connectivity_;
    std::shared_ptr<torch::nn::Module> synapse_;
    ConnectOptions options_;
    std::vector<std::shared_ptr<PlasticityRule>> plasticity_rules_;
};

// ---------------------------------------------------------------------------
// Free function: connect two populations.
//
// This is the primary user-facing API for wiring populations together.
// It delegates to the TopologyGenerator to produce the block structure
// and masks, wraps the result in a Connection, and returns it.
// ---------------------------------------------------------------------------
std::shared_ptr<Connection> connect(
    std::shared_ptr<Population> source,
    std::shared_ptr<Population> target,
    TopologyGenerator& topology,
    std::shared_ptr<torch::nn::Module> synapse,
    const ConnectOptions& options = {});

} // namespace nrn
