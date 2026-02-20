#pragma once

#include <torch/torch.h>
#include <nrn/core/options.h>

namespace nrn {
namespace neuron {

// ============================================================================
// LIF (Leaky Integrate-and-Fire) Options
// ============================================================================
// All values in SI units: volts, seconds, farads, siemens, amperes.
//
// Default parameters correspond to a standard cortical neuron:
//   v_rest    = -65 mV   (resting membrane potential)
//   v_thresh  = -50 mV   (spike threshold)
//   v_reset   = -65 mV   (post-spike reset potential)
//   tau_m     = 10 ms    (membrane time constant)
//   tau_ref   = 2 ms     (absolute refractory period)
//   c_m       = 250 pF   (membrane capacitance)
//   i_bg      = 0 A      (background current)
// ============================================================================
struct LIFOptions {
    NRN_ARG(double, v_rest,    -0.065);    // V
    NRN_ARG(double, v_thresh,  -0.050);    // V
    NRN_ARG(double, v_reset,   -0.065);    // V
    NRN_ARG(double, tau_m,      0.010);    // s
    NRN_ARG(double, tau_ref,    0.002);    // s
    NRN_ARG(double, c_m,        250e-12); // F
    NRN_ARG(double, i_bg,       0.0);      // A  (constant background current)
    NRN_ARG(torch::Dtype, dtype, torch::kFloat32);
};

// ============================================================================
// AdEx (Adaptive Exponential Integrate-and-Fire) Options
// ============================================================================
// Default parameters from Brette & Gerstner (2005), SI units.
//
//   v_rest    = -70.6 mV
//   v_thresh  = -50.4 mV  (effective threshold for exponential term)
//   v_reset   = -70.6 mV
//   v_peak    = 0 mV      (spike cutoff / peak)
//   tau_m     = 9.37 ms   (membrane time constant, tau_m = c_m / g_l)
//   tau_w     = 144 ms    (adaptation time constant)
//   tau_ref   = 2 ms      (absolute refractory period)
//   c_m       = 281 pF    (membrane capacitance)
//   g_l       = 30 nS     (leak conductance)
//   a         = 4 nS      (subthreshold adaptation coupling)
//   b         = 0.0805 nA (spike-triggered adaptation increment)
//   delta_t   = 2 mV      (slope factor of exponential term)
//   i_bg      = 0 A       (constant background current)
// ============================================================================
struct AdExOptions {
    NRN_ARG(double, v_rest,    -0.0706);   // V
    NRN_ARG(double, v_thresh,  -0.0504);   // V
    NRN_ARG(double, v_reset,   -0.0706);   // V
    NRN_ARG(double, v_peak,     0.0);      // V
    NRN_ARG(double, tau_m,      9.37e-3);  // s
    NRN_ARG(double, tau_w,      0.144);    // s
    NRN_ARG(double, tau_ref,    0.002);    // s
    NRN_ARG(double, c_m,        281e-12);  // F
    NRN_ARG(double, g_l,        30e-9);    // S
    NRN_ARG(double, a,          4e-9);     // S
    NRN_ARG(double, b,          80.5e-12); // A  (0.0805 nA)
    NRN_ARG(double, delta_t,    0.002);    // V
    NRN_ARG(double, i_bg,       0.0);      // A
    NRN_ARG(torch::Dtype, dtype, torch::kFloat32);
};

// ============================================================================
// Izhikevich Options
// ============================================================================
// Dimensionless convention following Izhikevich (2003).
//
// Dynamics:
//   dv/dt = 0.04*v^2 + 5*v + 140 - u + I
//   du/dt = a * (b*v - u)
//   if v >= v_peak: v = c, u = u + d
//
// Default: Regular Spiking (RS) neuron.
//   a = 0.02, b = 0.2, c = -65, d = 8, v_peak = 30
// ============================================================================
struct IzhikevichOptions {
    NRN_ARG(double, a,       0.02);  // Recovery time scale (dimensionless)
    NRN_ARG(double, b,       0.2);   // Recovery sensitivity to v (dimensionless)
    NRN_ARG(double, c,      -65.0);  // Post-spike reset of v (mV-like, dimensionless)
    NRN_ARG(double, d,       8.0);   // Post-spike increment of u (dimensionless)
    NRN_ARG(double, v_peak, 30.0);   // Spike cutoff (mV-like, dimensionless)
    NRN_ARG(double, v_init, -65.0);  // Initial membrane potential (dimensionless)
    NRN_ARG(double, u_init,   0.0);  // Initial recovery variable (dimensionless)
    NRN_ARG(torch::Dtype, dtype, torch::kFloat32);
};

} // namespace neuron
} // namespace nrn
