/// @file device.h
/// @brief Convenience factory for torch::Device placement.
///
/// @details
/// Wraps @c torch::Device construction behind a simple static-method API
/// so callers can write @c nrn::Device::GPU(0) or @c nrn::Device::Default()
/// without spelling out @c torch::Device internals.

#pragma once

#include <torch/torch.h>

namespace nrn {

/// @brief Convenience factory for torch::Device placement.
///
/// @details
/// All methods are @c static — @c Device is not instantiated.
struct Device {
    /// @brief Return a CUDA device for the given index.
    ///
    /// @param index  GPU ordinal (0-based, default 0).
    /// @return @c torch::Device on CUDA.
    static torch::Device GPU(int index = 0) {
        return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(index));
    }

    /// @brief Return the CPU device.
    /// @return @c torch::Device on CPU.
    static torch::Device CPU() {
        return torch::Device(torch::kCPU);
    }

    /// @brief Return the default device: GPU 0 if a CUDA device is available, else CPU.
    /// @return Appropriate @c torch::Device.
    static torch::Device Default() {
        return torch::cuda::is_available() ? GPU(0) : CPU();
    }

    /// @brief Return the number of available CUDA GPUs.
    /// @return Non-negative integer; 0 on CPU-only builds.
    static int64_t gpu_count() {
        return torch::cuda::device_count();
    }
};

} // namespace nrn
