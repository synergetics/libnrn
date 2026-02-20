#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace nrn {

// ---------------------------------------------------------------------------
// BlockIndex — CSR-of-dense-blocks sparse structure.
//
// Encodes which (source_block, target_block) pairs have dense connectivity
// blocks allocated.  The data layout mirrors CSR (Compressed Sparse Row):
//
//   row_ptr  [n_target_blocks + 1]  — start offset into col_idx per target block row
//   col_idx  [n_blocks]             — source block column index for each block
//
// This is the "Level 0" structure from the masked-dense paradigm: it says
// which population-pair blocks exist, while the dense data within each
// block is stored in the ConnectivityTensor weight/mask tensors.
// ---------------------------------------------------------------------------
struct BlockIndex {
    torch::Tensor row_ptr;  ///< [n_target_blocks + 1], int32
    torch::Tensor col_idx;  ///< [n_blocks], int32

    /// Number of dense blocks.
    int64_t n_blocks() const { return col_idx.size(0); }

    /// Number of target block rows.
    int64_t n_rows() const {
        return row_ptr.size(0) > 0 ? row_ptr.size(0) - 1 : 0;
    }

    /// Find the linear index of the block at (target_block_row, source_block_col).
    /// Returns -1 if the block does not exist.
    int64_t find_block(int64_t target_row, int64_t source_col) const;
};

// ---------------------------------------------------------------------------
// ConnectivityTensor — block-dense + mask connectivity representation.
//
// Stores the full connectivity between a source and target population
// using the masked-dense paradigm:
//
//     effective = weights .* structural_mask .* modulatory_mask
//     I_target += effective @ spike_source
//
// All per-synapse tensors are indexed as [n_blocks, B, B] where B is the
// block_size.
// ---------------------------------------------------------------------------
struct ConnectivityTensor {
    // -- Block structure ---------------------------------------------------
    BlockIndex block_index;
    int64_t block_size   = 0;
    int64_t n_source     = 0;     ///< Total source neurons.
    int64_t n_target     = 0;     ///< Total target neurons.

    // -- Per-block dense data [n_blocks, B, B] -----------------------------
    torch::Tensor weights;            ///< Synaptic efficacy (float).
    torch::Tensor structural_mask;    ///< Binary: which connections exist.
    torch::Tensor modulatory_mask;    ///< Continuous gain (neuromodulatory).
    torch::Tensor delays;             ///< Axonal delay in timesteps (int32).

    // -- Optional per-synapse state ----------------------------------------
    /// Arbitrary named synapse state tensors (e.g., STP variables u, x).
    std::unordered_map<std::string, torch::Tensor> synapse_state;

    /// Pre- and post-synaptic eligibility traces for plasticity.
    std::optional<torch::Tensor> trace_pre;
    std::optional<torch::Tensor> trace_post;

    // -- Methods -----------------------------------------------------------

    /// Compute effective connectivity: weights .* structural .* modulatory.
    torch::Tensor effective_weights() const;

    /// Move all tensors to the given device.
    void to(torch::Device device);

    /// Total number of possible synapses (n_blocks * B * B).
    int64_t total_capacity() const;

    /// Number of active (structurally existing) synapses.
    int64_t active_count() const;

    /// Connection density: active_count / total_capacity.
    double density() const;
};

} // namespace nrn
