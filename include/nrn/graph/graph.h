#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/core/state.h>
#include <nrn/connectivity/connectivity_tensor.h>
#include <nrn/graph/edge_types.h>
#include <nrn/plasticity/plasticity_rule.h>

namespace nrn {

// Forward declarations.
struct Region;

// ---------------------------------------------------------------------------
// NodeGroup — a contiguous block of nodes in the execution graph.
//
// Maps 1:1 to a Population in Phase 1. In future phases, a single neuron
// (multi-compartment) may be represented as multiple NodeGroups with
// different neuron_group_id values.
// ---------------------------------------------------------------------------
struct NodeGroup {
    uint32_t id;
    std::string name;           ///< Population name.
    NrnModule module;           ///< Copy of Population::module.
    State* state;               ///< NON-OWNING pointer to Population::state.
    int64_t offset;             ///< Start index in conceptual global node array.
    int64_t count;              ///< Number of nodes.
    int64_t neuron_group_id;    ///< -1 for point neurons; shared for compartmental.
    bool is_soma;               ///< true = emits spikes.
};

// ---------------------------------------------------------------------------
// EdgeGroup — connectivity between two NodeGroups.
//
// Wraps either a dense matrix or the existing block-sparse ConnectivityTensor,
// plus the synapse model and plasticity rules for this edge.
// ---------------------------------------------------------------------------
struct EdgeGroup {
    uint32_t id;
    uint32_t source_group_id;   ///< Index into Graph::node_groups.
    uint32_t target_group_id;
    EdgeTrigger trigger;
    EdgeRepresentation representation;  ///< Actual (resolved from Auto).

    // --- Block-sparse storage (active when representation == BlockSparse) ---
    ConnectivityTensor* block_sparse;   ///< NON-OWNING pointer to Connection's tensor.

    // --- Dense storage (active when representation == Dense) ---
    torch::Tensor dense_weights;         ///< [N_tgt, N_src]
    torch::Tensor dense_structural_mask; ///< [N_tgt, N_src] (optional)
    torch::Tensor dense_modulatory_mask; ///< [N_tgt, N_src] (optional)

    NrnModule synapse;
    std::vector<PlasticityRule> plasticity_rules;

    // Plasticity traces for dense representation.
    std::optional<torch::Tensor> dense_trace_pre;
    std::optional<torch::Tensor> dense_trace_post;
};

// ---------------------------------------------------------------------------
// Graph — the internal execution representation.
//
// Compiled from a Region via graph_compile(). The simulation loop operates
// on the Graph rather than directly on Population/Connection/Region.
// ---------------------------------------------------------------------------
struct Graph {
    std::vector<NodeGroup> node_groups;
    std::vector<EdgeGroup> edge_groups;
    int64_t total_nodes = 0;
    torch::Device device = torch::kCPU;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compile a Region into an execution Graph.
///
/// Node groups map 1:1 to populations. Edge groups map 1:1 to connections.
/// Auto representation is resolved based on density and size heuristics.
/// State pointers are non-owning — the Region must outlive the Graph.
Graph* graph_compile(Region* region, torch::Device device = torch::kCPU);

/// Destroy a Graph.
void graph_destroy(Graph* graph);

/// Find a node group by population name. Returns nullptr if not found.
NodeGroup* graph_find_node_group(Graph* graph, const std::string& name);

} // namespace nrn
