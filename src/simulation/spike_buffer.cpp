#include <nrn/simulation/spike_buffer.h>

namespace nrn {

SpikeBuffer::SpikeBuffer(int64_t n,
                         int64_t max_delay_steps,
                         torch::Device device)
    : n_(n),
      max_delay_steps_(max_delay_steps),
      write_pos_(0) {
    TORCH_CHECK(n > 0, "SpikeBuffer: n must be positive");
    TORCH_CHECK(max_delay_steps > 0,
                "SpikeBuffer: max_delay_steps must be positive");

    buffer_ = torch::zeros({max_delay_steps, n},
                           torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(device));
}

void SpikeBuffer::push(const torch::Tensor& spikes) {
    TORCH_CHECK(spikes.dim() == 1 && spikes.size(0) == n_,
                "SpikeBuffer::push: expected 1-D tensor of size ", n_,
                ", got shape ", spikes.sizes());

    // Write into the current slot (modular ring index).
    int64_t slot = write_pos_ % max_delay_steps_;
    buffer_[slot].copy_(spikes);
    ++write_pos_;
}

torch::Tensor SpikeBuffer::read(int64_t delay_steps) const {
    TORCH_CHECK(delay_steps >= 1 && delay_steps <= max_delay_steps_,
                "SpikeBuffer::read: delay_steps must be in [1, ",
                max_delay_steps_, "], got ", delay_steps);

    // If no data has been pushed yet, the buffer is all zeros.
    // Reading any valid delay returns a zero vector, which is correct for
    // simulation initialization (no spikes at t < 0).
    if (write_pos_ < static_cast<uint64_t>(delay_steps)) {
        return torch::zeros({n_}, buffer_.options());
    }

    // The most-recently-written slot is (write_pos_ - 1).
    // `delay_steps` ago is (write_pos_ - delay_steps).
    int64_t idx = static_cast<int64_t>((write_pos_ - delay_steps) %
                                        static_cast<uint64_t>(max_delay_steps_));
    return buffer_[idx];
}

torch::Tensor SpikeBuffer::read_batch(const torch::Tensor& delay_steps) const {
    TORCH_CHECK(delay_steps.dim() == 1 && delay_steps.size(0) == n_,
                "SpikeBuffer::read_batch: expected 1-D int64 tensor of size ",
                n_, ", got shape ", delay_steps.sizes());

    // Compute row indices: (write_pos_ - delay_steps) % max_delay_steps_
    // delay_steps is int64 on whatever device the buffer is on.
    auto delays = delay_steps.to(torch::kLong).to(buffer_.device());

    auto wp = torch::full({n_}, static_cast<int64_t>(write_pos_),
                          torch::TensorOptions()
                              .dtype(torch::kLong)
                              .device(buffer_.device()));

    // Row indices into the ring buffer.
    auto row_idx = (wp - delays) % max_delay_steps_;
    // Column indices: 0 .. N-1.
    auto col_idx = torch::arange(n_,
                                 torch::TensorOptions()
                                     .dtype(torch::kLong)
                                     .device(buffer_.device()));

    return buffer_.index({row_idx, col_idx});
}

void SpikeBuffer::reset() {
    buffer_.zero_();
    write_pos_ = 0;
}

} // namespace nrn
