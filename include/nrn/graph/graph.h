/// @file graph.h
/// @brief Core execution-graph types compiled from a Region.
///
/// @details
/// The simulation loop operates on Graph, NodeGroup, and EdgeGroup rather
/// than the higher-level Population / Connection / Region objects.
/// A Graph is produced by graph_compile() and must be destroyed with
/// graph_destroy().  All State pointers inside NodeGroups are non-owning —
/// **the source Region must outlive the Graph**.
///
/// @see graph_compile(), graph_step()

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

/// @brief A contiguous block of neurons (nodes) in the execution graph.
///
/// @details
/// Maps 1:1 to a Population in Phase 1.  In future phases, a single
/// multi-compartment neuron may be split into several NodeGroups sharing the
/// same neuron_group_id, with only the soma NodeGroup (is_soma == @c true)
/// emitting spikes.
///
/// NodeGroup::state is a non-owning pointer into the parent Population —
/// the Region must outlive the Graph.
///
/// @see EdgeGroup, Graph, graph_compile()
struct NodeGroup {
    uint32_t id;                ///< Unique index within Graph::node_groups.
    std::string name;           ///< Population name.
    NrnModule module;           ///< Copy of Population::module (ops-table handle).
    State* state;               ///< Non-owning pointer to Population::state.
    int64_t offset;             ///< Start index in the conceptual global node array.
    int64_t count;              ///< Number of neurons in this group.
    int64_t neuron_group_id;    ///< Shared across compartments of one neuron; @c -1 for point neurons.
    bool is_soma;               ///< @c true if this group emits spikes (soma compartment).
};

/// @brief Directed connectivity between two NodeGroups.
///
/// @details
/// Wraps either a dense weight matrix or a block-sparse ConnectivityTensor,
/// together with the synapse model and any plasticity rules governing this
/// connection.
///
/// Exactly one storage variant is active at runtime, determined by
/// EdgeGroup::representation (never @c Auto after graph_compile()).
///
/// @see NodeGroup, Graph, EdgeTrigger, EdgeRepresentation, graph_compile()
struct EdgeGroup {
    uint32_t id;                                  ///< Unique index within Graph::edge_groups.
    uint32_t source_group_id;                     ///< Index into Graph::node_groups for the source.
    uint32_t target_group_id;                     ///< Index into Graph::node_groups for the target.
    EdgeTrigger trigger;                          ///< Spike-driven or continuous delivery.
    EdgeRepresentation representation;            ///< Resolved storage format (never @c Auto at runtime).

    // --- Block-sparse storage (active when representation == BlockSparse) ---
    ConnectivityTensor* block_sparse;             ///< Non-owning pointer to the Connection's ConnectivityTensor.

    // --- Dense storage (active when representation == Dense) ---
    torch::Tensor dense_weights;                  ///< Weight matrix @c [N_tgt, N_src].
    torch::Tensor dense_structural_mask;          ///< Binary structural mask @c [N_tgt, N_src] (may be empty).
    torch::Tensor dense_modulatory_mask;          ///< Floating-point modulatory mask @c [N_tgt, N_src] (may be empty).

    NrnModule synapse;                            ///< Synapse model applied after the weight multiply.
    std::vector<PlasticityRule> plasticity_rules; ///< Ordered list of plasticity rules (may be empty).

    // Plasticity eligibility traces for the dense representation.
    std::optional<torch::Tensor> dense_trace_pre;  ///< Pre-synaptic trace @c [N_src].
    std::optional<torch::Tensor> dense_trace_post; ///< Post-synaptic trace @c [N_tgt].
};

/// @brief The compiled execution representation of a Region.
///
/// @details
/// Produced by graph_compile() from a Region.  The simulation loop
/// (graph_step()) operates on this structure rather than on the higher-level
/// objects.
///
/// All State pointers held by NodeGroups are non-owning; the source Region
/// must remain alive for the lifetime of the Graph.
///
/// @see graph_compile(), graph_destroy(), graph_step()
struct Graph {
    std::vector<NodeGroup> node_groups; ///< All neuron groups, indexed by NodeGroup::id.
    std::vector<EdgeGroup> edge_groups; ///< All connectivity groups, indexed by EdgeGroup::id.
    int64_t total_nodes = 0;            ///< Total neuron count across all NodeGroups.
    torch::Device device = torch::kCPU; ///< Device on which all tensors reside.
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// @brief Compile a Region into an execution Graph.
///
/// @details
///   - NodeGroups are created 1:1 from Populations.
///   - EdgeGroups are created 1:1 from Connections.
///   - @c Auto representation is resolved via the @c should_use_dense()
///     heuristic (density > 30% and N_src * N_tgt < 10 M → Dense; otherwise
///     BlockSparse).
///   - State pointers are non-owning: **the Region must outlive the returned
///     Graph**.
///
/// @param region  Source Region to compile; must not be null.
/// @param device  Target device for tensor allocation (default: CPU).
/// @return        Heap-allocated Graph; caller must call graph_destroy() when done.
///
/// @see graph_destroy(), Graph
Graph* graph_compile(Region* region, torch::Device device = torch::kCPU);

/// @brief Destroy a Graph produced by graph_compile().
///
/// @param graph  Graph to destroy; may be null (no-op).
///
/// @see graph_compile()
void graph_destroy(Graph* graph);

/// @brief Find a NodeGroup by population name.
///
/// @param graph  Graph to search; must not be null.
/// @param name   Population name to look up.
/// @return       Pointer to the matching NodeGroup, or @c nullptr if not found.
///
/// @see NodeGroup::name
NodeGroup* graph_find_node_group(Graph* graph, const std::string& name);

} // namespace nrn
