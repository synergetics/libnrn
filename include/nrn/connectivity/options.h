/// @file options.h
/// @brief Options for creating connections between populations.
///
/// @details
/// ConnectOptions controls the block size, storage representation,
/// weight scaling, delay handling, and trigger type for a connection.
///
/// @see connect(), Connection, ConnectivityRepresentation

#pragma once

#include <torch/torch.h>

#include <nrn/core/options.h>
#include <nrn/graph/edge_types.h>

namespace nrn {

/// @brief Storage strategy for a ConnectivityTensor.
///
/// @see ConnectOptions::representation
enum class ConnectivityRepresentation {
    BlockDense,   ///< Block-dense with mask overlays (default).
    Sparse,       ///< Literal CSR sparse (fallback for very sparse patterns).
    Procedural,   ///< On-the-fly computed connectivity (extreme scale, planned).
};

// Shorthand constants for ConnectivityRepresentation.
inline constexpr auto BlockDense  = ConnectivityRepresentation::BlockDense;  ///< @see ConnectivityRepresentation::BlockDense
inline constexpr auto Sparse      = ConnectivityRepresentation::Sparse;      ///< @see ConnectivityRepresentation::Sparse
inline constexpr auto Procedural  = ConnectivityRepresentation::Procedural;  ///< @see ConnectivityRepresentation::Procedural

/// @brief Builder-pattern options for creating a Connection.
///
/// @see connect(), Connection
struct ConnectOptions {
    NRN_ARG(int64_t, block_size, 256);  ///< Side length @c B of each dense connectivity block.
    NRN_ARG(ConnectivityRepresentation, representation, BlockDense); ///< Storage strategy.
    NRN_ARG(double, weight_scale, 1.0);                              ///< Global weight scaling factor.
    NRN_ARG(bool, with_delays, true);                                ///< Whether to store per-synapse delays.
    NRN_ARG(double, default_delay, 0.001);                           ///< Default axonal delay in seconds.
    NRN_ARG(torch::Dtype, weight_dtype, torch::kFloat32);            ///< Data type for weight tensors.
    NRN_ARG(torch::Dtype, mask_dtype,   torch::kFloat16);            ///< Data type for mask tensors.
    NRN_ARG(EdgeTrigger, trigger, EdgeTrigger::Spike);               ///< Delivery trigger (spike or continuous).
};

} // namespace nrn
