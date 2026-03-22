/// @file options.h
/// @brief Builder-pattern helper macro for options structs.
///
/// @details
/// Defines the @ref NRN_ARG macro used throughout the library to declare
/// named, typed option fields with defaults and a fluent setter interface.
/// Options structs using this macro follow the builder pattern:
/// @code
///   LIFOptions opts;
///   opts.v_rest(-0.070).tau_m(0.015).v_thresh(-0.050);
/// @endcode

#pragma once

#include <torch/torch.h>

/// @brief Declare a builder-pattern option field inside an options struct.
///
/// @details
/// Generates three things inside the enclosing struct:
///   1. A private-ish data member @c name##_ with the given default value.
///   2. A @c const getter returning @c name##_.
///   3. A fluent setter returning @c *this by reference so calls can be chained.
///
/// Example:
/// @code
///   struct MyOptions {
///       NRN_ARG(double, tau_m, 0.010);  // 10 ms
///   };
///   MyOptions opts;
///   opts.tau_m(0.020);          // set
///   double t = opts.tau_m();    // get
/// @endcode
///
/// @param type         C++ type of the option (e.g. @c double, @c bool).
/// @param name         Name of the option (used for getter/setter and member).
/// @param default_val  Default value for the member.
#define NRN_ARG(type, name, default_val)                                      \
    type name##_ = default_val;                                               \
    auto name() const -> type { return name##_; }                             \
    auto name(type val) -> decltype(*this)& { name##_ = val; return *this; }
