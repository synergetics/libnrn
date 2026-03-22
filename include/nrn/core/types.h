/// @file types.h
/// @brief Fundamental scalar typedefs and SI unit-literal operators.
///
/// @details
/// All quantities in libnrn are expressed in SI base units:
///   - Time / Duration in **seconds** (s)
///   - Voltage in **volts** (V)
///   - Current in **amperes** (A)
///   - Capacitance in **farads** (F)
///   - Conductance in **siemens** (S)
///   - Distance in **metres** (m)
///   - Concentration in **mol/L**
///
/// The @ref nrn::literals namespace provides user-defined literal operators
/// for convenient notation (e.g. @c 0.1_ms, @c -70.6_mV).  Opt in with:
/// @code
///   using namespace nrn::literals;
/// @endcode

#pragma once

#include <cstdint>

namespace nrn {

/// @brief Simulation time in seconds (SI).
using Time = double;

/// @brief Duration (interval) in seconds (SI).
using Duration = double;

/// @brief User-defined literal operators for convenient SI unit specification.
///
/// @details
/// All literals convert their argument to the SI base unit (seconds, volts,
/// amperes, farads, siemens, metres).  Opt in with:
/// @code
///   using namespace nrn::literals;
/// @endcode
namespace literals {

// -- Time / Duration --
constexpr Duration operator""_s(long double v)   { return static_cast<double>(v); }        ///< Seconds → seconds.
constexpr Duration operator""_ms(long double v)  { return static_cast<double>(v) * 1e-3; } ///< Milliseconds → seconds.
constexpr Duration operator""_us(long double v)  { return static_cast<double>(v) * 1e-6; } ///< Microseconds → seconds.

// -- Voltage --
constexpr double operator""_V(long double v)     { return static_cast<double>(v); }        ///< Volts → volts.
constexpr double operator""_mV(long double v)    { return static_cast<double>(v) * 1e-3; } ///< Millivolts → volts.

// -- Current --
constexpr double operator""_A(long double v)     { return static_cast<double>(v); }         ///< Amperes → amperes.
constexpr double operator""_nA(long double v)    { return static_cast<double>(v) * 1e-9; }  ///< Nanoamperes → amperes.
constexpr double operator""_pA(long double v)    { return static_cast<double>(v) * 1e-12; } ///< Picoamperes → amperes.

// -- Capacitance --
constexpr double operator""_F(long double v)     { return static_cast<double>(v); }         ///< Farads → farads.
constexpr double operator""_pF(long double v)    { return static_cast<double>(v) * 1e-12; } ///< Picofarads → farads.

// -- Conductance --
constexpr double operator""_S(long double v)     { return static_cast<double>(v); }        ///< Siemens → siemens.
constexpr double operator""_nS(long double v)    { return static_cast<double>(v) * 1e-9; } ///< Nanosiemens → siemens.

// -- Distance --
constexpr double operator""_m(long double v)     { return static_cast<double>(v); }        ///< Metres → metres.
constexpr double operator""_um(long double v)    { return static_cast<double>(v) * 1e-6; } ///< Micrometres → metres.

// -- Concentration --
constexpr double operator""_uM(long double v)    { return static_cast<double>(v) * 1e-6; } ///< Micromolar → mol/L.
constexpr double operator""_nM(long double v)    { return static_cast<double>(v) * 1e-9; } ///< Nanomolar → mol/L.

} // namespace literals
} // namespace nrn
