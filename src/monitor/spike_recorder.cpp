#include <nrn/monitor/spike_recorder.h>

#include <algorithm>
#include <utility>

namespace nrn {

SpikeRecorder::SpikeRecorder(std::string pop_name, int64_t subsample)
    : pop_name_(std::move(pop_name)),
      subsample_(subsample) {
    TORCH_CHECK(subsample_ >= 1,
                "SpikeRecorder: subsample must be >= 1, got ", subsample_);
}

void SpikeRecorder::record(const State& state, Time t) {
    ++call_count_;
    if ((call_count_ % subsample_) != 0) {
        return;
    }

    if (!state.contains("spike")) {
        return;
    }

    // Spike tensor: 1-D float [N], nonzero entries are spikes.
    auto spikes = state.get("spike").to(torch::kCPU).to(torch::kFloat32);
    auto nonzero = spikes.nonzero().squeeze(1); // [K] indices

    if (nonzero.numel() == 0) {
        return;
    }

    auto ids_accessor = nonzero.accessor<int64_t, 1>();
    for (int64_t i = 0; i < ids_accessor.size(0); ++i) {
        neuron_ids_.push_back(static_cast<float>(ids_accessor[i]));
        times_.push_back(static_cast<float>(t));
    }
}

void SpikeRecorder::reset() {
    neuron_ids_.clear();
    times_.clear();
    call_count_ = 0;
}

torch::Tensor SpikeRecorder::get_spikes() const {
    int64_t k = static_cast<int64_t>(neuron_ids_.size());
    if (k == 0) {
        return torch::zeros({0, 2}, torch::kFloat32);
    }

    auto result = torch::empty({k, 2}, torch::kFloat32);
    auto acc = result.accessor<float, 2>();
    for (int64_t i = 0; i < k; ++i) {
        acc[i][0] = neuron_ids_[static_cast<size_t>(i)];
        acc[i][1] = times_[static_cast<size_t>(i)];
    }
    return result;
}

torch::Tensor SpikeRecorder::get_spike_times(int64_t neuron_id) const {
    std::vector<float> matched;
    float target = static_cast<float>(neuron_id);
    for (size_t i = 0; i < neuron_ids_.size(); ++i) {
        if (neuron_ids_[i] == target) {
            matched.push_back(times_[i]);
        }
    }
    if (matched.empty()) {
        return torch::zeros({0}, torch::kFloat32);
    }
    return torch::from_blob(matched.data(),
                            {static_cast<int64_t>(matched.size())},
                            torch::kFloat32)
        .clone();
}

int64_t SpikeRecorder::spike_count() const {
    return static_cast<int64_t>(neuron_ids_.size());
}

} // namespace nrn
