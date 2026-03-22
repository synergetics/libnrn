/// @file options.h
/// @brief Parameter option structs for plasticity rules.
///
/// @see STDPState, stdp_create()

#pragma once

#include <nrn/core/options.h>

namespace nrn {

/// @brief Parameters for pair-based Spike-Timing-Dependent Plasticity (STDP).
///
/// @details
/// Implements the classic asymmetric STDP rule:
/// @code
///   dw = A_plus  * pre_trace  * post_spike   // LTP: pre fires before post
///      - A_minus * post_trace * pre_spike     // LTD: post fires before pre
///
///   d(trace_pre)  / dt = -trace_pre  / tau_plus
///   d(trace_post) / dt = -trace_post / tau_minus
///   on pre  spike: trace_pre  += 1
///   on post spike: trace_post += 1
/// @endcode
/// Weight updates are gated by structural and modulatory masks and clamped
/// to @c [w_min, w_max].
///
/// @see STDPState, stdp_create()
struct STDPOptions {
    NRN_ARG(double, tau_plus,      0.020);  ///< Pre-synaptic trace decay time constant (s).
    NRN_ARG(double, tau_minus,     0.020);  ///< Post-synaptic trace decay time constant (s).
    NRN_ARG(double, a_plus,        0.01);   ///< LTP amplitude.
    NRN_ARG(double, a_minus,      -0.012);  ///< LTD amplitude (must be negative).
    NRN_ARG(double, w_max,         1.0);    ///< Maximum allowable weight.
    NRN_ARG(double, w_min,         0.0);    ///< Minimum allowable weight.
    NRN_ARG(double, learning_rate, 1.0);    ///< Global learning rate multiplier.
};

} // namespace nrn
