#pragma once

#include <cstdint>

#include <nrn/core/types.h>

namespace nrn {

/// Multi-rate simulation clock with fast / slow / structural tiers.
///
/// The clock tracks a single monotonic simulation time and exposes
/// boundary queries so the simulation runner can decide when to invoke
/// slow-timescale subsystems (astrocytes, neuromodulation) and structural
/// plasticity updates.
///
/// Clock is a plain C++ value type — it is not a Module and carries no
/// GPU tensors.
class Clock {
public:
    /// Default constructor — zero-initialized, must be reassigned before use.
    Clock() : dt_fast_(0), dt_slow_(0), dt_structural_(0),
              slow_ratio_(1), structural_ratio_(1) {}

    /// Construct a clock with the three timescale durations.
    ///
    /// `dt_slow` and `dt_structural` must be exact integer multiples of
    /// `dt_fast`.
    Clock(Duration dt_fast, Duration dt_slow, Duration dt_structural);

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Current simulation time in seconds.
    Time time() const { return current_time_; }

    /// Current fast-step index (0-based, monotonically increasing).
    uint64_t step() const { return fast_step_; }

    /// Timescale durations.
    Duration dt_fast() const { return dt_fast_; }
    Duration dt_slow() const { return dt_slow_; }
    Duration dt_structural() const { return dt_structural_; }

    // ------------------------------------------------------------------
    // Advance
    // ------------------------------------------------------------------

    /// Advance by one fast timestep.
    void advance_fast();

    // ------------------------------------------------------------------
    // Boundary queries (call *after* advance_fast)
    // ------------------------------------------------------------------

    /// True if we have just crossed a slow-clock boundary.
    bool is_slow_boundary() const;

    /// True if we have just crossed a structural-clock boundary.
    bool is_structural_boundary() const;

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    /// Reset time and step counter to zero.
    void reset();

private:
    Duration dt_fast_;
    Duration dt_slow_;
    Duration dt_structural_;

    Time current_time_ = 0.0;
    uint64_t fast_step_ = 0;

    // Pre-computed integer ratios for boundary checks.
    uint64_t slow_ratio_;        // dt_slow / dt_fast
    uint64_t structural_ratio_;  // dt_structural / dt_fast
};

} // namespace nrn
