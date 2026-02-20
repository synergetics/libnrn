#pragma once

#include <nrn/core/options.h>

namespace nrn {

/// Options for pair-based Spike-Timing-Dependent Plasticity (STDP).
///
/// The classic asymmetric STDP rule:
///
///     dw = A_plus  * x_pre  * post_spike   (LTP, pre before post)
///        - |A_minus| * x_post * pre_spike   (LTD, post before pre)
///
///     dx_pre  / dt = -x_pre  / tau_plus
///     dx_post / dt = -x_post / tau_minus
///
/// On each pre spike:  x_pre  += 1
/// On each post spike: x_post += 1
///
/// Weight updates are gated by structural and modulatory masks and
/// clamped to [w_min, w_max].
struct STDPOptions {
    NRN_ARG(double, tau_plus, 0.020);        ///< Pre-trace time constant (s).
    NRN_ARG(double, tau_minus, 0.020);       ///< Post-trace time constant (s).
    NRN_ARG(double, a_plus, 0.01);           ///< LTP amplitude.
    NRN_ARG(double, a_minus, -0.012);        ///< LTD amplitude (negative).
    NRN_ARG(double, w_max, 1.0);             ///< Maximum weight.
    NRN_ARG(double, w_min, 0.0);             ///< Minimum weight.
    NRN_ARG(double, learning_rate, 1.0);     ///< Global learning rate scaling.
};

} // namespace nrn
