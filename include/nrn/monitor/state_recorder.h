/// @file state_recorder.h
/// @brief Recorder that samples continuous state variables every (sub)step.
///
/// @details
/// On each @c state_recorder_record() call, the recorder clones the
/// requested state tensors and appends them to per-variable history lists.
/// The full history can be retrieved as a stacked @c [T, N] tensor via
/// @c state_recorder_get().
///
/// @see StateRecorderState, state_recorder_create(), state_recorder_as_recorder()

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

/// @brief Internal state for the continuous state-variable recorder.
///
/// @see state_recorder_create(), state_recorder_as_recorder()
struct StateRecorderState {
    std::string pop_name;                      ///< Name of the observed population.
    std::vector<std::string> variables;        ///< Variables to record (e.g. @c {"v", "w"}).
    int64_t subsample  = 1;                    ///< Record every @c subsample-th call.
    int64_t call_count = 0;                    ///< Number of @c record() calls so far.

    /// Per-variable history: maps variable name → list of per-timestep @c [N] tensor snapshots.
    std::unordered_map<std::string, std::vector<torch::Tensor>> history;

    std::vector<float> times; ///< Recorded simulation timestamps (s).
};

/// @name Free functions — StateRecorderState
/// @{

/// @brief Allocate and return a new StateRecorderState.
///
/// @param pop_name   Name of the observed population.
/// @param variables  State variable names to track (e.g. @c {"v", "w"}).
/// @param subsample  Record only every @p subsample-th call.
/// @return Heap-allocated state; caller must call @c state_recorder_destroy().
StateRecorderState* state_recorder_create(std::string pop_name,
                                          std::vector<std::string> variables,
                                          int64_t subsample = 1);

/// @brief Free a StateRecorderState.
/// @param s  Pointer to destroy; may be null (no-op).
void state_recorder_destroy(StateRecorderState* s);

/// @brief Clone tracked variables from @p state and append to history.
/// @param self   Opaque pointer to a StateRecorderState.
/// @param state  Named tensor bag of the observed population.
/// @param t      Current simulation time (s).
void state_recorder_record(void* self, const State& state, double t);

/// @brief Clear all stored history and timestamps.
/// @param self  Opaque pointer to a StateRecorderState.
void state_recorder_reset(void* self);

/// @brief Return the population name (C string; for @c recorder_ops compatibility).
/// @param self  Opaque pointer to a StateRecorderState.
const char* state_recorder_population_name(void* self);

/// @brief Retrieve the recorded history for one variable as a @c [T, N] tensor.
///
/// @param s         Recorder state.
/// @param variable  Name of the state variable to retrieve.
/// @return Tensor of shape @c [T, N] (T = sample count, N = neuron count) on CPU.
torch::Tensor state_recorder_get(const StateRecorderState* s,
                                 const std::string& variable);

/// @brief Return all recording timestamps as a 1-D @c float tensor @c [T].
/// @param s  Recorder state.
/// @return Monotonically increasing simulation times at which samples were taken.
torch::Tensor state_recorder_get_times(const StateRecorderState* s);

/// @brief Names of the state variables being recorded.
/// @param s  Recorder state.
/// @return Reference to the variable name list.
const std::vector<std::string>& state_recorder_variables(
    const StateRecorderState* s);

/// @}

extern recorder_ops state_recorder_ops; ///< Ops table for StateRecorderState.

/// @brief Wrap a StateRecorderState into a type-erased Recorder handle.
/// @param s  Pointer to an existing StateRecorderState.
/// @return Recorder with @c state_recorder_ops dispatch table.
inline Recorder state_recorder_as_recorder(StateRecorderState* s) {
    return Recorder{static_cast<void*>(s), &state_recorder_ops};
}

} // namespace nrn
