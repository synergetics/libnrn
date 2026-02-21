#pragma once

#include <memory>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>
#include <nrn/connectivity/connectivity_tensor.h>
#include <nrn/connectivity/options.h>

namespace nrn {

// Forward declarations.
struct Population;
struct PlasticityRule;
struct TopologyGenerator;

// ---------------------------------------------------------------------------
// Connection — owns connectivity between a source and target population.
// ---------------------------------------------------------------------------
struct Connection {
    std::shared_ptr<Population> source;
    std::shared_ptr<Population> target;
    ConnectivityTensor connectivity;
    NrnModule synapse;      // type-erased synapse module (may be unused)
    ConnectOptions options;
    std::vector<PlasticityRule> plasticity_rules;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Create a connection from pre-built components.
Connection* connection_create(std::shared_ptr<Population> source,
                              std::shared_ptr<Population> target,
                              ConnectivityTensor connectivity,
                              NrnModule synapse,
                              ConnectOptions options);

void connection_destroy(Connection* conn);

/// Deliver presynaptic spikes through the connectivity to produce
/// postsynaptic currents.
void connection_deliver(Connection* conn, const torch::Tensor& spikes,
                        double t, double dt);

/// Attach a plasticity rule.
void connection_attach(Connection* conn, PlasticityRule rule);

/// Run all attached plasticity rules for one update step.
void connection_update_plasticity(Connection* conn,
                                  const State& pre_state,
                                  const State& post_state,
                                  double t, double dt);

// ---------------------------------------------------------------------------
// Free function: connect two populations.
// ---------------------------------------------------------------------------
std::shared_ptr<Connection> connect(
    std::shared_ptr<Population> source,
    std::shared_ptr<Population> target,
    TopologyGenerator* topology,
    NrnModule synapse,
    const ConnectOptions& options = {});

} // namespace nrn
