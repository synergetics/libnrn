#pragma once

#include <string>

#include <nrn/core/state.h>
#include <nrn/core/types.h>

namespace nrn {

/// Ops table for recorders.
///
/// A Recorder observes a population's State at each timestep and
/// accumulates data for later retrieval.  Concrete implementations
/// handle spikes, continuous state variables, or aggregate statistics.
struct recorder_ops {
    /// Record relevant data from state at simulation time t.
    ///
    /// Implementations decide internally whether to actually store data
    /// on this call (e.g. based on a subsample interval).
    void (*record)(void* self, const State& state, double t);

    /// Clear all accumulated data.
    void (*reset)(void* self);

    /// Name of the population this recorder is attached to.
    const char* (*population_name)(void* self);
};

/// Type-erased recorder handle.
///
/// Holds an opaque pointer to the concrete recorder state and a pointer
/// to the ops table that dispatches to the correct implementation.
struct Recorder {
    void* impl;
    recorder_ops* ops;
};

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

inline void recorder_record(Recorder* r, const State& state, double t) {
    r->ops->record(r->impl, state, t);
}

inline void recorder_reset(Recorder* r) {
    r->ops->reset(r->impl);
}

inline const char* recorder_population_name(Recorder* r) {
    return r->ops->population_name(r->impl);
}

} // namespace nrn
