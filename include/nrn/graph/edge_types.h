/// @file edge_types.h
/// @brief Enumerations describing how EdgeGroups trigger and store synaptic connectivity.
///
/// @details
/// Two orthogonal axes characterise every EdgeGroup in the execution graph:
///   - @ref EdgeTrigger        — *when* current flows (spike-gated vs. continuous).
///   - @ref EdgeRepresentation — *how* the weight matrix is stored (dense vs. block-sparse).
///
/// @see EdgeGroup, graph_compile()

#pragma once

namespace nrn {

/// @brief Controls when synaptic current is delivered from a source NodeGroup to a target NodeGroup.
///
/// @details
/// Each EdgeGroup has exactly one trigger type, evaluated during Phase 2
/// of the graph_step() simulation loop.
///
/// @see graph_step(), EdgeGroup::trigger
enum class EdgeTrigger {

    /// @brief Spike-driven (chemical) synapse.
    ///
    /// @details
    /// Current is delivered only when presynaptic neurons fire:
    /// @code
    ///   I_syn[tgt] += W @ spikes[src]
    /// @endcode
    /// where @c spikes is a binary float tensor (@c 1.0 = fired, @c 0.0 = silent),
    /// read from the delay-aware spike buffer or directly from state.
    /// In the dense representation this reduces to a single @c torch::mm,
    /// mapping directly to tensor-core operations.
    ///
    /// This is the default trigger for Connection objects.
    ///
    /// @see edge_group_deliver_spike()
    Spike,

    /// @brief Continuous (electrical / gap-junction) synapse.
    ///
    /// @details
    /// Current flows every timestep, proportional to the voltage difference
    /// between each pre/post neuron pair:
    /// @code
    ///   I[i] = sum_j  g[i,j] * (V[j]_src - V[i]_tgt)
    /// @endcode
    /// In the dense representation this is computed as:
    /// @code
    ///   I_tgt += W @ V_src - rowsum(W) * V_tgt
    /// @endcode
    /// avoiding an explicit per-element loop.  Net current is conservative:
    /// total injected current sums to zero across coupled populations.
    ///
    /// @see edge_group_deliver_continuous()
    Continuous,
};

/// @brief Selects the storage format for the weight / connectivity matrix of an EdgeGroup.
///
/// @details
/// The library implements the *masked-dense paradigm*:
/// @code
///   effective = W & M_structural & M_modulatory & M_stochastic
///   output    = f(effective @ input)
/// @endcode
/// This converts irregular sparse biological graphs into regular dense tensor
/// operations that GPUs and tensor cores are optimised for.  The two concrete
/// formats trade memory footprint for compute regularity.
///
/// @see EdgeGroup, graph_compile(), should_use_dense()
enum class EdgeRepresentation {

    /// @brief Full dense weight matrix of shape @c [N_tgt, N_src].
    ///
    /// @details
    /// Stored in EdgeGroup::dense_weights (plus optional
    /// EdgeGroup::dense_structural_mask and EdgeGroup::dense_modulatory_mask).
    /// Spike delivery reduces to a single @c torch::mm; continuous delivery
    /// adds a row-sum correction term.
    ///
    /// Selected automatically by graph_compile() when:
    ///   - connection density > 30%, **and**
    ///   - @c N_src * N_tgt < 10,000,000 (approximately 40 MB at float32).
    ///
    /// @note Dense-native plasticity is planned for Phase 2 and not yet supported.
    Dense,

    /// @brief Block-sparse storage via ConnectivityTensor (CSR-of-dense-blocks).
    ///
    /// @details
    /// Three-level hierarchy mirroring biological organisation:
    ///   - **Level 0** — BlockIndex (CSR): which population-pair blocks exist
    ///     (analogous to sparse inter-region connectivity).
    ///   - **Level 1** — weights @c [n_blocks, B, B]: dense data per block
    ///     (analogous to intra-region population connectivity).
    ///   - **Level 2** — mask overlays (structural, modulatory): applied
    ///     element-wise within each block (local circuit detail).
    ///
    /// EdgeGroup::block_sparse holds a non-owning pointer to the Connection's
    /// ConnectivityTensor — **the Region must outlive the Graph**.
    /// Delivery iterates block-by-block, calling @c torch::mm per block.
    /// Plasticity is fully supported via ConnectivityTensor update operations.
    ///
    /// This is the default representation; preferred for large or sparse networks.
    BlockSparse,

    /// @brief Automatic selection by graph_compile().
    ///
    /// @details
    /// Never stored at runtime.  graph_compile() resolves this to either
    /// Dense or BlockSparse via the @c should_use_dense() heuristic:
    ///   - density > 30% **and** total entries < 10 M → Dense
    ///   - otherwise → BlockSparse
    Auto,
};

} // namespace nrn
