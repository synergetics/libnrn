/// @file recorder.h
/// @brief Type-erased Recorder handle and recorder_ops dispatch table.
///
/// @details
/// A Recorder observes a Population's State at each (or every @c n-th)
/// fast timestep and accumulates data for later retrieval.
/// Concrete implementations (SpikeRecorderState, StateRecorderState)
/// provide a @c recorder_ops table and wrap themselves via @c *_as_recorder().
///
/// @see SpikeRecorderState, StateRecorderState, graph_step()

#pragma once

#include <string>

#include <nrn/core/state.h>
#include <nrn/core/types.h>

namespace nrn {

/// @brief Ops table for recorders.
///
/// @details
/// Every concrete recorder type provides a static instance of this struct.
struct recorder_ops {
    /// @brief Sample relevant data from @p state at simulation time @p t.
    ///
    /// @details
    /// Implementations decide internally whether to store data on this call
    /// (e.g. based on a subsample interval counter).
    ///
    /// @param self   Opaque pointer to the concrete recorder state.
    /// @param state  Named tensor bag of the observed population.
    /// @param t      Current simulation time (s).
    void (*record)(void* self, const State& state, double t);

    /// @brief Clear all accumulated data.
    /// @param self  Opaque pointer to the concrete recorder state.
    void (*reset)(void* self);

    /// @brief Return the name of the population this recorder is attached to.
    /// @param self  Opaque pointer to the concrete recorder state.
    /// @return Null-terminated C string; valid for the lifetime of the recorder.
    const char* (*population_name)(void* self);
};

/// @brief Type-erased recorder handle.
///
/// @details
/// Holds an opaque @c void* to the concrete recorder state and a pointer to
/// its ops table.  Wrap concrete recorders with @c spike_recorder_as_recorder()
/// or @c state_recorder_as_recorder().
///
/// @see recorder_ops, spike_recorder_as_recorder(), state_recorder_as_recorder()
struct Recorder {
    void* impl;           ///< Opaque pointer to the concrete recorder struct.
    recorder_ops* ops;    ///< Dispatch table.
};

/// @brief Sample state from @p state at time @p t.
/// @see recorder_ops::record
inline void recorder_record(Recorder* r, const State& state, double t) {
    r->ops->record(r->impl, state, t);
}

/// @brief Clear all recorded data.
/// @see recorder_ops::reset
inline void recorder_reset(Recorder* r) {
    r->ops->reset(r->impl);
}

/// @brief Return the name of the population this recorder is attached to.
/// @see recorder_ops::population_name
inline const char* recorder_population_name(Recorder* r) {
    return r->ops->population_name(r->impl);
}

} // namespace nrn
