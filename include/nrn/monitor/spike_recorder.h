/// @file spike_recorder.h
/// @brief Recorder that accumulates discrete spike events (neuron index + time).
///
/// @details
/// On each call to @c spike_recorder_record(), the recorder looks for a
/// tensor named @c "spike" in the State (a 1-D float tensor where nonzero
/// entries indicate spikes).  Firing neuron indices and the current time are
/// appended to CPU-side vectors.  The full event history can be retrieved
/// as a @c [K, 2] tensor via @c spike_recorder_get_spikes().
///
/// @see SpikeRecorderState, spike_recorder_create(), spike_recorder_as_recorder()

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/monitor/recorder.h>

namespace nrn {

/// @brief Internal state for the spike event recorder.
///
/// @see spike_recorder_create(), spike_recorder_as_recorder()
struct SpikeRecorderState {
    std::string pop_name;              ///< Name of the observed population.
    int64_t subsample  = 1;            ///< Record every @c subsample-th call.
    int64_t call_count = 0;            ///< Number of @c record() calls so far.

    std::vector<float> neuron_ids;     ///< CPU-side list of firing neuron indices.
    std::vector<float> times;          ///< CPU-side list of corresponding spike times (s).
};

/// @name Free functions — SpikeRecorderState
/// @{

/// @brief Allocate and return a new SpikeRecorderState.
///
/// @param pop_name   Name of the observed population.
/// @param subsample  Record only every @p subsample-th @c record() call.
/// @return Heap-allocated state; caller must call @c spike_recorder_destroy().
SpikeRecorderState* spike_recorder_create(std::string pop_name,
                                          int64_t subsample = 1);

/// @brief Free a SpikeRecorderState.
/// @param s  Pointer to destroy; may be null (no-op).
void spike_recorder_destroy(SpikeRecorderState* s);

/// @brief Scan @p state for the @c "spike" tensor and append firing events.
/// @param self   Opaque pointer to a SpikeRecorderState.
/// @param state  Named tensor bag of the observed population.
/// @param t      Current simulation time (s).
void spike_recorder_record(void* self, const State& state, double t);

/// @brief Clear all stored spike data.
/// @param self  Opaque pointer to a SpikeRecorderState.
void spike_recorder_reset(void* self);

/// @brief Return the population name (C string; for @c recorder_ops compatibility).
/// @param self  Opaque pointer to a SpikeRecorderState.
const char* spike_recorder_population_name(void* self);

/// @brief Return all recorded spikes as a @c [K, 2] float tensor.
///
/// @details
/// Column 0 is the neuron index (float), column 1 is the spike time (s).
/// Returns an empty @c [0, 2] tensor if no spikes have been recorded.
///
/// @param s  Recorder state.
/// @return @c [K, 2] tensor on CPU.
torch::Tensor spike_recorder_get_spikes(const SpikeRecorderState* s);

/// @brief Return all spike times for a single neuron.
///
/// @param s          Recorder state.
/// @param neuron_id  Zero-based neuron index.
/// @return 1-D float tensor of spike times (s).
torch::Tensor spike_recorder_get_spike_times(const SpikeRecorderState* s,
                                             int64_t neuron_id);

/// @brief Total number of spike events accumulated so far.
/// @param s  Recorder state.
/// @return Number of individual spike events.
int64_t spike_recorder_spike_count(const SpikeRecorderState* s);

/// @}

extern recorder_ops spike_recorder_ops; ///< Ops table for SpikeRecorderState.

/// @brief Wrap a SpikeRecorderState into a type-erased Recorder handle.
/// @param s  Pointer to an existing SpikeRecorderState.
/// @return Recorder with @c spike_recorder_ops dispatch table.
inline Recorder spike_recorder_as_recorder(SpikeRecorderState* s) {
    return Recorder{static_cast<void*>(s), &spike_recorder_ops};
}

} // namespace nrn
