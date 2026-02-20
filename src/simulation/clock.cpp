#include <nrn/simulation/clock.h>

#include <cmath>
#include <stdexcept>

namespace nrn {

Clock::Clock(Duration dt_fast, Duration dt_slow, Duration dt_structural)
    : dt_fast_(dt_fast),
      dt_slow_(dt_slow),
      dt_structural_(dt_structural),
      current_time_(0.0),
      fast_step_(0) {
    // Validate positive durations.
    if (dt_fast_ <= 0.0) {
        throw std::invalid_argument("dt_fast must be positive");
    }
    if (dt_slow_ <= 0.0) {
        throw std::invalid_argument("dt_slow must be positive");
    }
    if (dt_structural_ <= 0.0) {
        throw std::invalid_argument("dt_structural must be positive");
    }

    // Compute integer ratios.  We round to the nearest integer and then
    // verify that the ratio is exact (within floating-point tolerance).
    double slow_r = dt_slow_ / dt_fast_;
    double struct_r = dt_structural_ / dt_fast_;

    slow_ratio_ = static_cast<uint64_t>(std::round(slow_r));
    structural_ratio_ = static_cast<uint64_t>(std::round(struct_r));

    if (slow_ratio_ == 0) {
        throw std::invalid_argument(
            "dt_slow must be >= dt_fast");
    }
    if (structural_ratio_ == 0) {
        throw std::invalid_argument(
            "dt_structural must be >= dt_fast");
    }

    // Check that the ratios are (close to) exact integers.
    constexpr double tol = 1e-6;
    if (std::abs(slow_r - static_cast<double>(slow_ratio_)) > tol) {
        throw std::invalid_argument(
            "dt_slow must be an integer multiple of dt_fast");
    }
    if (std::abs(struct_r - static_cast<double>(structural_ratio_)) > tol) {
        throw std::invalid_argument(
            "dt_structural must be an integer multiple of dt_fast");
    }
}

void Clock::advance_fast() {
    ++fast_step_;
    current_time_ = static_cast<double>(fast_step_) * dt_fast_;
}

bool Clock::is_slow_boundary() const {
    return (fast_step_ % slow_ratio_) == 0;
}

bool Clock::is_structural_boundary() const {
    return (fast_step_ % structural_ratio_) == 0;
}

void Clock::reset() {
    fast_step_ = 0;
    current_time_ = 0.0;
}

} // namespace nrn
