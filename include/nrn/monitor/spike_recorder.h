#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/monitor/recorder.h>

namespace nrn {

/// Internal state for the spike event recorder.
///
/// On each call to record, the recorder looks for a tensor named
/// "spike" in the State (a 1-D float tensor where nonzero entries
/// indicate spikes).  When spikes are found, their neuron indices and
/// the current time are appended to internal buffers.
struct SpikeRecorderState {
    std::string pop_name;
    int64_t subsample = 1;
    int64_t call_count = 0;

    // Accumulated spike data on CPU.
    std::vector<float> neuron_ids;
    std::vector<float> times;
};

// ---------------------------------------------------------------------------
// Free functions operating on SpikeRecorderState
// ---------------------------------------------------------------------------

/// Allocate and return a new SpikeRecorderState on the heap.
///
/// @param pop_name   Name of the population being recorded.
/// @param subsample  Record only every subsample-th call.
SpikeRecorderState* spike_recorder_create(std::string pop_name,
                                          int64_t subsample = 1);

/// Free a SpikeRecorderState allocated with spike_recorder_create.
void spike_recorder_destroy(SpikeRecorderState* s);

/// Scan the state for spikes and store events.
void spike_recorder_record(void* self, const State& state, double t);

/// Clear all stored spike data.
void spike_recorder_reset(void* self);

/// Population name (returns C string for ops table compatibility).
const char* spike_recorder_population_name(void* self);

/// Return all recorded spikes as a [K, 2] float tensor.
///
/// Column 0 is the neuron index (float), column 1 is the spike time.
/// Returns an empty [0, 2] tensor if no spikes have been recorded.
torch::Tensor spike_recorder_get_spikes(const SpikeRecorderState* s);

/// Return spike times for a single neuron.
torch::Tensor spike_recorder_get_spike_times(const SpikeRecorderState* s,
                                             int64_t neuron_id);

/// Total number of spike events recorded so far.
int64_t spike_recorder_spike_count(const SpikeRecorderState* s);

/// Ops table for SpikeRecorderState.
extern recorder_ops spike_recorder_ops;

/// Wrap a SpikeRecorderState pointer into a type-erased Recorder handle.
inline Recorder spike_recorder_as_recorder(SpikeRecorderState* s) {
    return Recorder{static_cast<void*>(s), &spike_recorder_ops};
}

} // namespace nrn
