/// @file options.h
/// @brief Parameter option structs for all built-in synapse models.
///
/// @details
/// All time constants and delays are in seconds (SI).  Weights are
/// dimensionless scaling factors.  Each struct uses @ref NRN_ARG for
/// fluent setter chaining.
///
/// @see StaticSynapse, ExponentialSynapse, DoubleExponentialSynapse, AMPASynapse

#pragma once

#include <nrn/core/options.h>

namespace nrn {

/// @brief Parameters for a static (fixed-weight, no temporal dynamics) synapse.
/// @see synapse::StaticSynapse
struct StaticSynapseOptions {
    NRN_ARG(double, weight, 1.0);   ///< Synaptic weight (dimensionless scaling factor).
    NRN_ARG(double, delay, 0.001);  ///< Axonal transmission delay (s).
};

/// @brief Parameters for a single-exponential decay synapse.
///
/// @details
/// @code
///   dI/dt = -I / tau_decay
///   I    += weight * spike  (on presynaptic spike)
/// @endcode
///
/// @see synapse::ExponentialSynapse
struct ExponentialSynapseOptions {
    NRN_ARG(double, tau_decay, 0.005); ///< Decay time constant (s).
    NRN_ARG(double, weight, 1.0);      ///< Synaptic weight.
    NRN_ARG(double, delay, 0.001);     ///< Axonal transmission delay (s).
};

/// @brief Parameters for a double-exponential (rise + decay) synapse.
///
/// @details
/// @code
///   I = I_decay - I_rise
///   dI_rise  / dt = -I_rise  / tau_rise
///   dI_decay / dt = -I_decay / tau_decay
///   on spike:  I_rise += weight;  I_decay += weight
/// @endcode
///
/// @see synapse::DoubleExponentialSynapse
struct DoubleExponentialSynapseOptions {
    NRN_ARG(double, tau_rise,  0.0005); ///< Rise time constant (s).
    NRN_ARG(double, tau_decay, 0.005);  ///< Decay time constant (s).
    NRN_ARG(double, weight, 1.0);       ///< Synaptic weight.
    NRN_ARG(double, delay, 0.001);      ///< Axonal transmission delay (s).
};

/// @brief Parameters for an AMPA receptor synapse with reversal potential.
///
/// @details
/// Conductance-based double-exponential kinetics:
/// @code
///   g = weight * (s_decay - s_rise)
///   I_syn = g * (v_post - e_rev)
/// @endcode
///
/// @see synapse::AMPASynapse
struct AMPAOptions {
    NRN_ARG(double, tau_rise,  0.0005); ///< Rise time constant (s).
    NRN_ARG(double, tau_decay, 0.005);  ///< Decay time constant (s).
    NRN_ARG(double, e_rev, 0.0);        ///< Reversal potential (V); 0 V for AMPA.
    NRN_ARG(double, weight, 1.0);       ///< Synaptic weight (peak conductance scaling).
    NRN_ARG(double, delay, 0.001);      ///< Axonal transmission delay (s).
};

} // namespace nrn
