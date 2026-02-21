#pragma once

#include <torch/torch.h>

namespace nrn {

/// Convenience helpers for device placement.
///
/// These wrap torch::Device to provide a consistent interface
/// for specifying GPU/CPU placement.
struct Device {
    /// Return a torch::Device for GPU `index`.
    static torch::Device GPU(int index = 0) {
        return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(index));
    }

    /// Return the CPU device.
    static torch::Device CPU() {
        return torch::Device(torch::kCPU);
    }

    /// Return the default device (GPU 0 if available, else CPU).
    static torch::Device Default() {
        return torch::cuda::is_available() ? GPU(0) : CPU();
    }

    /// Return the number of available GPUs.
    static int64_t gpu_count() {
        return torch::cuda::device_count();
    }
};

} // namespace nrn
