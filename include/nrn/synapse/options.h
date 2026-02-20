#pragma once

#include <nrn/core/options.h>

namespace nrn {

/// Options for a static (fixed-weight) synapse.
struct StaticSynapseOptions {
    NRN_ARG(double, weight, 1.0);
    NRN_ARG(double, delay, 0.001);
};

/// Options for a single-exponential decay synapse.
struct ExponentialSynapseOptions {
    NRN_ARG(double, tau_decay, 0.005);
    NRN_ARG(double, weight, 1.0);
    NRN_ARG(double, delay, 0.001);
};

/// Options for a double-exponential (rise + decay) synapse.
struct DoubleExponentialSynapseOptions {
    NRN_ARG(double, tau_rise, 0.0005);
    NRN_ARG(double, tau_decay, 0.005);
    NRN_ARG(double, weight, 1.0);
    NRN_ARG(double, delay, 0.001);
};

/// Options for an AMPA receptor synapse with reversal potential.
struct AMPAOptions {
    NRN_ARG(double, tau_rise, 0.0005);
    NRN_ARG(double, tau_decay, 0.005);
    NRN_ARG(double, e_rev, 0.0);
    NRN_ARG(double, weight, 1.0);
    NRN_ARG(double, delay, 0.001);
};

} // namespace nrn
