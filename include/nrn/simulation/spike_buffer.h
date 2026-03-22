/// @file spike_buffer.h
/// @brief Dense ring-buffer for synaptic spike delay lines.
///
/// @details
/// Stores a binary (@c float 0/1) tensor of shape @c [max_delay_steps, N]
/// where @c N is the number of neurons.  Each call to @c push() writes the
/// current spike vector into the ring buffer at the write head and advances
/// the head.  @c read(delay) returns the spike vector that was pushed
/// @p delay_steps steps ago.
///
/// All storage lives on the specified @c torch::Device (typically CUDA).
///
/// @see Simulation, sim_create()

#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace nrn {

/// @brief Dense ring-buffer for spike delay lines.
///
/// @details
/// Internally stores a @c float tensor of shape @c [max_delay_steps, N].
/// Reads and writes are O(1) pointer arithmetic in the ring.
class SpikeBuffer {
public:
    /// @brief Construct a SpikeBuffer.
    ///
    /// @param n                Number of neurons.
    /// @param max_delay_steps  Maximum supported delay in fast timesteps.
    /// @param device           Torch device for the internal buffer tensor.
    SpikeBuffer(int64_t n, int64_t max_delay_steps, torch::Device device);

    /// @brief Push a spike vector for the current timestep.
    ///
    /// @details
    /// Writes @p spikes into the ring at the current write head and advances
    /// the head.  The oldest entry is silently overwritten.
    ///
    /// @param spikes  1-D @c float tensor of shape @c [N] with 0.0 or 1.0 entries.
    void push(const torch::Tensor& spikes);

    /// @brief Read the spike vector that was pushed @p delay_steps steps ago.
    ///
    /// @param delay_steps  How many steps back to look (1 = the most recent push).
    /// @return  1-D @c float tensor of shape @c [N].
    torch::Tensor read(int64_t delay_steps) const;

    /// @brief Batched read with per-neuron delay offsets.
    ///
    /// @details
    /// For each neuron @c i, reads the spike from @c delay_steps[i] steps ago.
    /// Useful when synaptic delays are heterogeneous across a population.
    ///
    /// @param delay_steps  1-D @c int64 tensor of shape @c [N] with per-neuron delays.
    /// @return  1-D @c float tensor of shape @c [N].
    torch::Tensor read_batch(const torch::Tensor& delay_steps) const;

    /// @brief Zero the entire buffer and reset the write head to position 0.
    void reset();

    /// @brief Maximum delay this buffer can represent (in fast timesteps).
    int64_t max_delay() const { return max_delay_steps_; }

    /// @brief Number of neurons tracked by this buffer.
    int64_t size() const { return n_; }

private:
    torch::Tensor buffer_;    ///< Ring storage @c [max_delay_steps, N].
    int64_t n_;               ///< Number of neurons.
    int64_t max_delay_steps_; ///< Ring depth.
    int64_t write_pos_ = 0;   ///< Next row to write into (wraps at @c max_delay_steps_).
};

} // namespace nrn
