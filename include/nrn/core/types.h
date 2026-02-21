#pragma once

#include <cstdint>

namespace nrn {

/// Simulation time in seconds (SI).
using Time = double;

/// Duration in seconds (SI).
using Duration = double;

/// User-defined literal operators for convenient unit specification.
///
/// All quantities are converted to SI base units (seconds, volts, amperes,
/// farads, siemens, metres, moles/litre).  These live in nrn::literals so
/// users can opt-in with `using namespace nrn::literals;`.
namespace literals {

// -- Time / Duration --
constexpr Duration operator""_s(long double v)   { return static_cast<double>(v); }
constexpr Duration operator""_ms(long double v)  { return static_cast<double>(v) * 1e-3; }
constexpr Duration operator""_us(long double v)  { return static_cast<double>(v) * 1e-6; }

// -- Voltage --
constexpr double operator""_V(long double v)     { return static_cast<double>(v); }
constexpr double operator""_mV(long double v)    { return static_cast<double>(v) * 1e-3; }

// -- Current --
constexpr double operator""_A(long double v)     { return static_cast<double>(v); }
constexpr double operator""_nA(long double v)    { return static_cast<double>(v) * 1e-9; }
constexpr double operator""_pA(long double v)    { return static_cast<double>(v) * 1e-12; }

// -- Capacitance --
constexpr double operator""_F(long double v)     { return static_cast<double>(v); }
constexpr double operator""_pF(long double v)    { return static_cast<double>(v) * 1e-12; }

// -- Conductance --
constexpr double operator""_S(long double v)     { return static_cast<double>(v); }
constexpr double operator""_nS(long double v)    { return static_cast<double>(v) * 1e-9; }

// -- Distance --
constexpr double operator""_m(long double v)     { return static_cast<double>(v); }
constexpr double operator""_um(long double v)    { return static_cast<double>(v) * 1e-6; }

// -- Concentration --
constexpr double operator""_uM(long double v)    { return static_cast<double>(v) * 1e-6; }
constexpr double operator""_nM(long double v)    { return static_cast<double>(v) * 1e-9; }

} // namespace literals
} // namespace nrn
