#include <nrn/monitor/spike_recorder.h>

#include <algorithm>
#include <utility>

namespace nrn {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

SpikeRecorderState* spike_recorder_create(std::string pop_name,
                                          int64_t subsample) {
    TORCH_CHECK(subsample >= 1,
                "spike_recorder_create: subsample must be >= 1, got ",
                subsample);

    auto* s = new SpikeRecorderState{};
    s->pop_name = std::move(pop_name);
    s->subsample = subsample;
    s->call_count = 0;
    return s;
}

void spike_recorder_destroy(SpikeRecorderState* s) {
    delete s;
}

// ---------------------------------------------------------------------------
// Ops implementations (void* self -> SpikeRecorderState*)
// ---------------------------------------------------------------------------

void spike_recorder_record(void* self, const State& state, double t) {
    auto* s = static_cast<SpikeRecorderState*>(self);

    ++s->call_count;
    if ((s->call_count % s->subsample) != 0) {
        return;
    }

    if (!state_contains(state, "spike")) {
        return;
    }

    // Spike tensor: 1-D float [N], nonzero entries are spikes.
    auto spikes = state_get(state, "spike").to(torch::kCPU).to(torch::kFloat32);
    auto nonzero = spikes.nonzero().squeeze(1); // [K] indices

    if (nonzero.numel() == 0) {
        return;
    }

    auto ids_accessor = nonzero.accessor<int64_t, 1>();
    for (int64_t i = 0; i < ids_accessor.size(0); ++i) {
        s->neuron_ids.push_back(static_cast<float>(ids_accessor[i]));
        s->times.push_back(static_cast<float>(t));
    }
}

void spike_recorder_reset(void* self) {
    auto* s = static_cast<SpikeRecorderState*>(self);
    s->neuron_ids.clear();
    s->times.clear();
    s->call_count = 0;
}

const char* spike_recorder_population_name(void* self) {
    auto* s = static_cast<SpikeRecorderState*>(self);
    return s->pop_name.c_str();
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

torch::Tensor spike_recorder_get_spikes(const SpikeRecorderState* s) {
    int64_t k = static_cast<int64_t>(s->neuron_ids.size());
    if (k == 0) {
        return torch::zeros({0, 2}, torch::kFloat32);
    }

    auto result = torch::empty({k, 2}, torch::kFloat32);
    auto acc = result.accessor<float, 2>();
    for (int64_t i = 0; i < k; ++i) {
        acc[i][0] = s->neuron_ids[static_cast<size_t>(i)];
        acc[i][1] = s->times[static_cast<size_t>(i)];
    }
    return result;
}

torch::Tensor spike_recorder_get_spike_times(const SpikeRecorderState* s,
                                             int64_t neuron_id) {
    std::vector<float> matched;
    float target = static_cast<float>(neuron_id);
    for (size_t i = 0; i < s->neuron_ids.size(); ++i) {
        if (s->neuron_ids[i] == target) {
            matched.push_back(s->times[i]);
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

int64_t spike_recorder_spike_count(const SpikeRecorderState* s) {
    return static_cast<int64_t>(s->neuron_ids.size());
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

recorder_ops spike_recorder_ops = {
    spike_recorder_record,
    spike_recorder_reset,
    spike_recorder_population_name,
};

} // namespace nrn
