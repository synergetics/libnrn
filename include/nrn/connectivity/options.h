#pragma once

#include <torch/torch.h>

#include <nrn/core/options.h>

namespace nrn {

/// Connectivity representation strategy.
enum class ConnectivityRepresentation {
    BlockDense,   ///< Block-dense with mask overlays (default).
    Sparse,       ///< Literal CSR sparse (fallback for very sparse patterns).
    Procedural,   ///< On-the-fly computed connectivity (extreme scale).
};

/// Shorthand constants.
inline constexpr auto BlockDense  = ConnectivityRepresentation::BlockDense;
inline constexpr auto Sparse      = ConnectivityRepresentation::Sparse;
inline constexpr auto Procedural  = ConnectivityRepresentation::Procedural;

/// Options for creating a Connection between two populations.
struct ConnectOptions {
    NRN_ARG(int64_t, block_size, 256);
    NRN_ARG(ConnectivityRepresentation, representation, BlockDense);
    NRN_ARG(double, weight_scale, 1.0);
    NRN_ARG(bool, with_delays, true);
    NRN_ARG(double, default_delay, 0.001);
    NRN_ARG(torch::Dtype, weight_dtype, torch::kFloat32);
    NRN_ARG(torch::Dtype, mask_dtype, torch::kFloat16);
};

} // namespace nrn
