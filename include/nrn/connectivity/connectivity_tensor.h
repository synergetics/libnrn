/// @file connectivity_tensor.h
/// @brief Block-sparse connectivity storage (CSR-of-dense-blocks).
///
/// @details
/// Implements the *masked-dense paradigm*:
/// @code
///   effective = weights .* structural_mask .* modulatory_mask
///   I_target += effective @ spike_source
/// @endcode
///
/// Two complementary structs:
///   - @ref BlockIndex — CSR row/col structure saying *which* blocks exist.
///   - @ref ConnectivityTensor — all per-synapse data indexed @c [n_blocks, B, B].
///
/// @see EdgeGroup, graph_compile()

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace nrn {

/// @brief CSR block-index encoding which (target_block, source_block) pairs exist.
///
/// @details
/// Mirrors standard CSR format but at block granularity:
/// @code
///   for target_row in 0..n_rows:
///       for k in row_ptr[target_row]..row_ptr[target_row+1]:
///           source_col = col_idx[k]
///           // block k is at (target_row, source_col)
/// @endcode
/// This is Level 0 of the three-level hierarchy: it records which
/// population-pair blocks have connectivity data allocated.
struct BlockIndex {
    torch::Tensor row_ptr;  ///< CSR row pointers @c [n_target_blocks + 1], int32.
    torch::Tensor col_idx;  ///< Source block column indices @c [n_blocks], int32.

    /// @brief Number of allocated dense blocks.
    int64_t n_blocks() const { return col_idx.size(0); }

    /// @brief Number of target block rows.
    int64_t n_rows() const {
        return row_ptr.size(0) > 0 ? row_ptr.size(0) - 1 : 0;
    }

    /// @brief Find the linear index of the block at (@p target_row, @p source_col).
    ///
    /// @param target_row  Target block row (0-based).
    /// @param source_col  Source block column (0-based).
    /// @return Linear block index, or @c -1 if the block does not exist.
    int64_t find_block(int64_t target_row, int64_t source_col) const;
};

/// @brief Block-dense + mask connectivity between a source and target population.
///
/// @details
/// All per-synapse tensors are indexed @c [n_blocks, B, B] where @c B is the
/// block size.  The effective weight used during spike delivery is:
/// @code
///   effective = weights .* structural_mask .* modulatory_mask
/// @endcode
///
/// Optional per-synapse state (STP variables, eligibility traces) is stored
/// in @c synapse_state and @c trace_pre / @c trace_post respectively.
///
/// @see BlockIndex, ConnectOptions, graph_compile()
struct ConnectivityTensor {
    // -- Block structure ---------------------------------------------------
    BlockIndex block_index; ///< CSR structure saying which blocks exist.
    int64_t block_size = 0; ///< Side length @c B of each dense block.
    int64_t n_source   = 0; ///< Total number of source (presynaptic) neurons.
    int64_t n_target   = 0; ///< Total number of target (postsynaptic) neurons.

    // -- Per-block dense data [n_blocks, B, B] -----------------------------
    torch::Tensor weights;          ///< Synaptic efficacies (float).
    torch::Tensor structural_mask;  ///< Binary: 1 = synapse exists, 0 = absent.
    torch::Tensor modulatory_mask;  ///< Continuous gain for neuromodulation (1.0 = neutral).
    torch::Tensor delays;           ///< Axonal delays in fast timesteps (int32).

    // -- Optional per-synapse state ----------------------------------------
    std::unordered_map<std::string, torch::Tensor> synapse_state; ///< Arbitrary named synapse state (e.g. STP u, x).
    std::optional<torch::Tensor> trace_pre;   ///< Pre-synaptic eligibility trace (for STDP etc.).
    std::optional<torch::Tensor> trace_post;  ///< Post-synaptic eligibility trace.

    // -- Methods -----------------------------------------------------------

    /// @brief Compute @c weights .* structural_mask .* modulatory_mask.
    /// @return Effective weight tensor @c [n_blocks, B, B].
    torch::Tensor effective_weights() const;

    /// @brief Move all tensors to @p device in place.
    void to(torch::Device device);

    /// @brief Total number of possible synapses: @c n_blocks * B * B.
    int64_t total_capacity() const;

    /// @brief Number of structurally active synapses (structural_mask.sum()).
    int64_t active_count() const;

    /// @brief Connection density: @c active_count() / @c total_capacity().
    double density() const;
};

} // namespace nrn
