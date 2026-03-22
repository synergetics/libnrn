/// @file options.h
/// @brief Parameter option structs for all built-in neuron models.
///
/// @details
/// All parameters are in SI base units (volts, seconds, farads, siemens,
/// amperes) unless otherwise noted.  Each struct uses @ref NRN_ARG to
/// provide fluent setter chaining:
/// @code
///   auto opts = nrn::neuron::LIFOptions{}
///       .v_rest(-0.070).tau_m(0.015);
/// @endcode
///
/// @see LIFNeuron, AdExNeuron, IzhikevichNeuron

#pragma once

#include <torch/torch.h>
#include <nrn/core/options.h>

namespace nrn {
namespace neuron {

/// @brief Parameters for the Leaky Integrate-and-Fire (LIF) neuron model.
///
/// @details
/// Default parameters correspond to a standard cortical point neuron:
/// | Parameter  | Default   | Unit | Description                        |
/// |------------|-----------|------|------------------------------------|
/// | v_rest     | -65 mV    | V    | Resting membrane potential         |
/// | v_thresh   | -50 mV    | V    | Spike threshold                    |
/// | v_reset    | -65 mV    | V    | Post-spike reset potential         |
/// | tau_m      |  10 ms    | s    | Membrane time constant             |
/// | tau_ref    |   2 ms    | s    | Absolute refractory period         |
/// | c_m        | 250 pF    | F    | Membrane capacitance               |
/// | i_bg       |   0 A     | A    | Constant background current        |
///
/// @see LIFNeuron, lif_create()
struct LIFOptions {
    NRN_ARG(double, v_rest,    -0.065);    ///< Resting membrane potential (V).
    NRN_ARG(double, v_thresh,  -0.050);    ///< Spike threshold (V).
    NRN_ARG(double, v_reset,   -0.065);    ///< Post-spike reset potential (V).
    NRN_ARG(double, tau_m,      0.010);    ///< Membrane time constant (s).
    NRN_ARG(double, tau_ref,    0.002);    ///< Absolute refractory period (s).
    NRN_ARG(double, c_m,        250e-12);  ///< Membrane capacitance (F).
    NRN_ARG(double, i_bg,       0.0);      ///< Constant background current (A).
    NRN_ARG(torch::Dtype, dtype, torch::kFloat32); ///< Tensor data type.
};

/// @brief Parameters for the Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
///
/// @details
/// Default parameters from Brette & Gerstner (2005), in SI units.
/// | Parameter  | Default    | Unit | Description                                 |
/// |------------|------------|------|---------------------------------------------|
/// | v_rest     | -70.6 mV   | V    | Resting membrane potential                  |
/// | v_thresh   | -50.4 mV   | V    | Effective threshold for exponential term    |
/// | v_reset    | -70.6 mV   | V    | Post-spike reset potential                  |
/// | v_peak     |    0 mV    | V    | Spike cutoff / voltage peak                 |
/// | tau_m      |  9.37 ms   | s    | Membrane time constant (c_m / g_l)          |
/// | tau_w      | 144 ms     | s    | Adaptation time constant                    |
/// | tau_ref    |   2 ms     | s    | Absolute refractory period                  |
/// | c_m        | 281 pF     | F    | Membrane capacitance                        |
/// | g_l        |  30 nS     | S    | Leak conductance                            |
/// | a          |   4 nS     | S    | Subthreshold adaptation coupling            |
/// | b          | 80.5 pA    | A    | Spike-triggered adaptation increment        |
/// | delta_t    |   2 mV     | V    | Slope factor of exponential term            |
/// | i_bg       |   0 A      | A    | Constant background current                 |
///
/// @see AdExNeuron, adex_create()
struct AdExOptions {
    NRN_ARG(double, v_rest,    -0.0706);   ///< Resting membrane potential (V).
    NRN_ARG(double, v_thresh,  -0.0504);   ///< Spike threshold (V).
    NRN_ARG(double, v_reset,   -0.0706);   ///< Post-spike reset potential (V).
    NRN_ARG(double, v_peak,     0.0);      ///< Spike cutoff voltage (V).
    NRN_ARG(double, tau_m,      9.37e-3);  ///< Membrane time constant (s).
    NRN_ARG(double, tau_w,      0.144);    ///< Adaptation time constant (s).
    NRN_ARG(double, tau_ref,    0.002);    ///< Absolute refractory period (s).
    NRN_ARG(double, c_m,        281e-12);  ///< Membrane capacitance (F).
    NRN_ARG(double, g_l,        30e-9);    ///< Leak conductance (S).
    NRN_ARG(double, a,          4e-9);     ///< Subthreshold adaptation (S).
    NRN_ARG(double, b,          80.5e-12); ///< Spike-triggered adaptation increment (A).
    NRN_ARG(double, delta_t,    0.002);    ///< Exponential slope factor (V).
    NRN_ARG(double, i_bg,       0.0);      ///< Constant background current (A).
    NRN_ARG(torch::Dtype, dtype, torch::kFloat32); ///< Tensor data type.
};

/// @brief Parameters for the Izhikevich neuron model (dimensionless convention).
///
/// @details
/// Follows the dimensionless convention from Izhikevich (2003):
/// @code
///   dv/dt = 0.04*v^2 + 5*v + 140 - u + I
///   du/dt = a * (b*v - u)
///   if v >= v_peak:  v = c,  u += d
/// @endcode
/// Default values produce a Regular Spiking (RS) cortical neuron:
/// | Parameter | Default | Description                               |
/// |-----------|---------|-------------------------------------------|
/// | a         | 0.02    | Recovery time scale                       |
/// | b         | 0.2     | Sensitivity of recovery variable to @c v |
/// | c         | -65     | Post-spike reset value of @c v            |
/// | d         | 8       | Post-spike increment of @c u              |
/// | v_peak    | 30      | Spike cutoff                              |
///
/// @note @c dt in @c izh_forward() is in seconds (SI); the implementation
///       converts to milliseconds internally to match the dimensionless units.
///
/// @see IzhikevichNeuron, izh_create()
struct IzhikevichOptions {
    NRN_ARG(double, a,       0.02);  ///< Recovery time scale (dimensionless).
    NRN_ARG(double, b,       0.2);   ///< Recovery sensitivity to @c v (dimensionless).
    NRN_ARG(double, c,      -65.0);  ///< Post-spike reset of @c v (dimensionless).
    NRN_ARG(double, d,       8.0);   ///< Post-spike increment of @c u (dimensionless).
    NRN_ARG(double, v_peak, 30.0);   ///< Spike cutoff (dimensionless).
    NRN_ARG(double, v_init, -65.0);  ///< Initial membrane potential (dimensionless).
    NRN_ARG(double, u_init,   0.0);  ///< Initial recovery variable (dimensionless).
    NRN_ARG(torch::Dtype, dtype, torch::kFloat32); ///< Tensor data type.
};

} // namespace neuron
} // namespace nrn
