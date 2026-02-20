#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nrn {

/// Dense ring-buffer for spike delay lines.
///
/// Stores a binary (float 0/1) tensor of shape `[max_delay_steps, N]` where
/// `N` is the number of neurons.  Each call to `push()` writes the current
/// spike vector into the ring buffer at the write head and advances the
/// head.  `read(delay)` returns the spike vector that was pushed `delay`
/// steps ago.
///
/// All storage lives on the specified torch::Device (typically CUDA).
class SpikeBuffer {
public:
    /// @param n                Number of neurons.
    /// @param max_delay_steps  Maximum supported delay in fast timesteps.
    /// @param device           Torch device for the internal buffer.
    SpikeBuffer(int64_t n, int64_t max_delay_steps, torch::Device device);

    /// Push a spike vector for the current timestep.
    ///
    /// @param spikes  A 1-D float tensor of shape [N] (0.0 or 1.0).
    void push(const torch::Tensor& spikes);

    /// Read the spike vector that was pushed `delay_steps` ago.
    ///
    /// @param delay_steps  How many steps back to look (1 = previous push).
    /// @return  A 1-D float tensor of shape [N].
    torch::Tensor read(int64_t delay_steps) const;

    /// Batched read: for each neuron, look back by a per-neuron delay.
    ///
    /// @param delay_steps  A 1-D int64 tensor of shape [N] giving per-neuron
    ///                     delays.
    /// @return  A 1-D float tensor of shape [N] — one spike value per neuron.
    torch::Tensor read_batch(const torch::Tensor& delay_steps) const;

    /// Zero the entire buffer and reset the write head.
    void reset();

    /// Maximum delay this buffer supports (in fast timesteps).
    int64_t max_delay() const { return max_delay_steps_; }

    /// Number of neurons tracked.
    int64_t size() const { return n_; }

private:
    torch::Tensor buffer_;    // [max_delay_steps, N]
    int64_t n_;
    int64_t max_delay_steps_;
    int64_t write_pos_ = 0;  // next row to write into
};

} // namespace nrn
