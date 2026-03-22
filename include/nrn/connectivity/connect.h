/// @file connect.h
/// @brief Connection struct and the connect() free function.
///
/// @details
/// A Connection owns the ConnectivityTensor, synapse model, and plasticity
/// rules for a directed edge between two Populations.  The high-level
/// connect() factory constructs a Connection from a TopologyGenerator.
///
/// @see ConnectivityTensor, connect(), TopologyGenerator, PlasticityRule

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

/// @brief Directed synaptic connection between two Populations.
///
/// @details
/// Owns the ConnectivityTensor and holds shared pointers to source and target
/// populations.  The synapse model is optional (may be a null NrnModule for
/// static connections).  Plasticity rules are evaluated by the simulation
/// loop at the appropriate timescale.
///
/// @see connect(), connection_deliver(), connection_update_plasticity()
struct Connection {
    std::shared_ptr<Population> source;  ///< Presynaptic population.
    std::shared_ptr<Population> target;  ///< Postsynaptic population.
    ConnectivityTensor connectivity;     ///< Block-dense weight / mask storage.
    NrnModule synapse;                   ///< Type-erased synapse model (may be unused).
    ConnectOptions options;              ///< Creation options.
    std::vector<PlasticityRule> plasticity_rules; ///< Ordered list of plasticity rules.
};

/// @brief Create a Connection from pre-built components.
///
/// @param source        Presynaptic population.
/// @param target        Postsynaptic population.
/// @param connectivity  Already-generated ConnectivityTensor.
/// @param synapse       Type-erased synapse module.
/// @param options       Creation options.
/// @return Heap-allocated Connection; caller takes ownership.
Connection* connection_create(std::shared_ptr<Population> source,
                              std::shared_ptr<Population> target,
                              ConnectivityTensor connectivity,
                              NrnModule synapse,
                              ConnectOptions options);

/// @brief Free a Connection.
/// @param conn  Pointer to the Connection; may be null (no-op).
void connection_destroy(Connection* conn);

/// @brief Deliver presynaptic spikes through the connectivity to produce postsynaptic currents.
///
/// @param conn    Connection to deliver through.
/// @param spikes  Binary float spike tensor for the source population.
/// @param t       Current simulation time (s).
/// @param dt      Timestep duration (s).
void connection_deliver(Connection* conn, const torch::Tensor& spikes,
                        double t, double dt);

/// @brief Attach a plasticity rule to the connection.
///
/// @param conn  Connection to modify.
/// @param rule  PlasticityRule to append (appended to plasticity_rules).
void connection_attach(Connection* conn, PlasticityRule rule);

/// @brief Run all attached plasticity rules for one update step.
///
/// @param conn       Connection whose rules are evaluated.
/// @param pre_state  State of the presynaptic population.
/// @param post_state State of the postsynaptic population.
/// @param t          Current simulation time (s).
/// @param dt         Timestep duration (s).
void connection_update_plasticity(Connection* conn,
                                  const State& pre_state,
                                  const State& post_state,
                                  double t, double dt);

/// @brief Connect two populations using a topology generator.
///
/// @details
/// Calls @c topology_generate() to produce a ConnectivityTensor, then
/// constructs a fully initialised Connection and adds it to @p target's
/// parent Region.
///
/// @param source    Presynaptic population.
/// @param target    Postsynaptic population.
/// @param topology  Generator that produces the connectivity pattern.
/// @param synapse   Type-erased synapse module.
/// @param options   Connection options (block size, delays, trigger type, etc.).
/// @return Shared pointer to the created Connection.
std::shared_ptr<Connection> connect(
    std::shared_ptr<Population> source,
    std::shared_ptr<Population> target,
    TopologyGenerator* topology,
    NrnModule synapse,
    const ConnectOptions& options = {});

} // namespace nrn
