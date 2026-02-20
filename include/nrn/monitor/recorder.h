#pragma once

#include <string>

#include <nrn/core/state.h>
#include <nrn/core/types.h>

namespace nrn {

/// Abstract base class for all recorders.
///
/// A Recorder observes a population's State at each timestep and
/// accumulates data for later retrieval.  Concrete subclasses handle
/// spikes, continuous state variables, or aggregate statistics.
class Recorder {
public:
    virtual ~Recorder() = default;

    /// Record relevant data from `state` at simulation time `t`.
    ///
    /// Implementations decide internally whether to actually store data
    /// on this call (e.g. based on a subsample interval).
    virtual void record(const State& state, Time t) = 0;

    /// Clear all accumulated data.
    virtual void reset() = 0;

    /// Name of the population this recorder is attached to.
    virtual const std::string& population_name() const = 0;
};

} // namespace nrn
