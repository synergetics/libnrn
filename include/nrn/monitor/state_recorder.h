#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/monitor/recorder.h>

namespace nrn {

/// Internal state for the continuous state variable recorder.
///
/// On each call to record, the recorder clones the requested state
/// tensors and appends them to per-variable histories.  The histories
/// can be retrieved as a stacked tensor of shape [T, N] via
/// state_recorder_get().
struct StateRecorderState {
    std::string pop_name;
    std::vector<std::string> variables;
    int64_t subsample = 1;
    int64_t call_count = 0;

    /// variable_name -> list of per-timestep snapshots (each [N]).
    std::unordered_map<std::string, std::vector<torch::Tensor>> history;

    /// Recorded timestamps.
    std::vector<float> times;
};

// ---------------------------------------------------------------------------
// Free functions operating on StateRecorderState
// ---------------------------------------------------------------------------

/// Allocate and return a new StateRecorderState on the heap.
///
/// @param pop_name   Name of the population being recorded.
/// @param variables  Names of state variables to track (e.g. {"v", "w"}).
/// @param subsample  Record only every subsample-th call.
StateRecorderState* state_recorder_create(std::string pop_name,
                                          std::vector<std::string> variables,
                                          int64_t subsample = 1);

/// Free a StateRecorderState allocated with state_recorder_create.
void state_recorder_destroy(StateRecorderState* s);

/// Clone tracked variables from state and store them.
void state_recorder_record(void* self, const State& state, double t);

/// Clear all stored history.
void state_recorder_reset(void* self);

/// Population name (returns C string for ops table compatibility).
const char* state_recorder_population_name(void* self);

/// Retrieve the recorded history for one variable.
///
/// @return Tensor of shape [T, N] (T = number of recorded samples,
///         N = number of neurons).
torch::Tensor state_recorder_get(const StateRecorderState* s,
                                 const std::string& variable);

/// Return the recording timestamps as a 1-D float tensor [T].
torch::Tensor state_recorder_get_times(const StateRecorderState* s);

/// Variable names being recorded.
const std::vector<std::string>& state_recorder_variables(
    const StateRecorderState* s);

/// Ops table for StateRecorderState.
extern recorder_ops state_recorder_ops;

/// Wrap a StateRecorderState pointer into a type-erased Recorder handle.
inline Recorder state_recorder_as_recorder(StateRecorderState* s) {
    return Recorder{static_cast<void*>(s), &state_recorder_ops};
}

} // namespace nrn
