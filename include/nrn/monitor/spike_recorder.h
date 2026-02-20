#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/monitor/recorder.h>

namespace nrn {

/// Records spike events as (neuron_id, time) pairs.
///
/// On each call to `record()`, the recorder looks for a tensor named
/// "spike" in the State (a 1-D float tensor where nonzero entries
/// indicate spikes).  When spikes are found, their neuron indices and
/// the current time are appended to internal buffers.
class SpikeRecorder : public Recorder {
public:
    /// @param pop_name   Name of the population being recorded.
    /// @param subsample  Record only every `subsample`-th call.
    explicit SpikeRecorder(std::string pop_name, int64_t subsample = 1);

    /// Scan the state for spikes and store events.
    void record(const State& state, Time t) override;

    /// Clear all stored spike data.
    void reset() override;

    /// Population name.
    const std::string& population_name() const override { return pop_name_; }

    /// Return all recorded spikes as a [K, 2] float tensor.
    ///
    /// Column 0 is the neuron index (float), column 1 is the spike time.
    /// Returns an empty [0, 2] tensor if no spikes have been recorded.
    torch::Tensor get_spikes() const;

    /// Return spike times for a single neuron.
    torch::Tensor get_spike_times(int64_t neuron_id) const;

    /// Total number of spike events recorded so far.
    int64_t spike_count() const;

private:
    std::string pop_name_;
    int64_t subsample_;
    int64_t call_count_ = 0;

    // Accumulated spike data on CPU.
    std::vector<float> neuron_ids_;
    std::vector<float> times_;
};

} // namespace nrn
