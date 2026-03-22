/// @file clock.h
/// @brief Multi-rate simulation clock with fast / slow / structural timescales.
///
/// @details
/// Clock tracks a single monotonic simulation time and exposes boundary
/// queries so graph_step() and the Simulation runner can decide when to
/// invoke slow-timescale subsystems (neuromodulation, astrocytes) and
/// structural plasticity updates.
///
/// @c dt_slow and @c dt_structural must be exact integer multiples of
/// @c dt_fast; this is checked at construction.
///
/// Clock is a plain value type — no GPU state, copyable.
///
/// @see SimulationOptions, graph_step()

#pragma once

#include <cstdint>

#include <nrn/core/types.h>

namespace nrn {

/// @brief Multi-rate simulation clock.
///
/// @details
/// Maintains:
///   - @c current_time_: wall-clock simulation time in seconds.
///   - @c fast_step_: monotonically increasing 0-based fast-tick counter.
///
/// Boundary queries (@c is_slow_boundary(), @c is_structural_boundary())
/// are evaluated via integer modular arithmetic on @c fast_step_ for
/// efficiency.
class Clock {
public:
    /// @brief Default constructor — zero-initialised; must be reassigned before use.
    Clock() : dt_fast_(0), dt_slow_(0), dt_structural_(0),
              slow_ratio_(1), structural_ratio_(1) {}

    /// @brief Construct a clock with the three timescale durations.
    ///
    /// @param dt_fast        Fast (neural dynamics) timestep in seconds.
    /// @param dt_slow        Slow (modulatory) timestep; must be an integer multiple of @p dt_fast.
    /// @param dt_structural  Structural plasticity timestep; must be an integer multiple of @p dt_fast.
    Clock(Duration dt_fast, Duration dt_slow, Duration dt_structural);

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// @brief Current simulation time in seconds.
    Time time() const { return current_time_; }

    /// @brief Current fast-step index (0-based, monotonically increasing).
    uint64_t step() const { return fast_step_; }

    Duration dt_fast()       const { return dt_fast_; }       ///< Fast timestep duration (s).
    Duration dt_slow()       const { return dt_slow_; }       ///< Slow timestep duration (s).
    Duration dt_structural() const { return dt_structural_; } ///< Structural timestep duration (s).

    // ------------------------------------------------------------------
    // Advance
    // ------------------------------------------------------------------

    /// @brief Advance by one fast timestep.
    void advance_fast();

    // ------------------------------------------------------------------
    // Boundary queries (call *after* advance_fast)
    // ------------------------------------------------------------------

    /// @brief Return @c true if the last @c advance_fast() crossed a slow-clock boundary.
    bool is_slow_boundary() const;

    /// @brief Return @c true if the last @c advance_fast() crossed a structural-clock boundary.
    bool is_structural_boundary() const;

    // ------------------------------------------------------------------
    // Reset
    // ------------------------------------------------------------------

    /// @brief Reset time and step counter to zero.
    void reset();

private:
    Duration dt_fast_;
    Duration dt_slow_;
    Duration dt_structural_;

    Time current_time_ = 0.0;
    uint64_t fast_step_ = 0;

    uint64_t slow_ratio_;        ///< dt_slow / dt_fast (integer).
    uint64_t structural_ratio_;  ///< dt_structural / dt_fast (integer).
};

} // namespace nrn
