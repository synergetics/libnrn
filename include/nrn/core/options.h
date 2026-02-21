#pragma once

#include <torch/torch.h>

/// Macro for declaring a builder-pattern option field.
///
/// Generates a private member `name##_` with type `type` and default value
/// `default_val`, plus a const getter and a fluent setter returning *this.
///
/// Usage inside an options struct:
///   NRN_ARG(double, v_rest, -0.065);
///
/// Expands to:
///   double v_rest_ = -0.065;
///   auto v_rest() const -> double { return v_rest_; }
///   auto v_rest(double val) -> Self& { v_rest_ = val; return *this; }
#define NRN_ARG(type, name, default_val)                                      \
    type name##_ = default_val;                                               \
    auto name() const -> type { return name##_; }                             \
    auto name(type val) -> decltype(*this)& { name##_ = val; return *this; }
