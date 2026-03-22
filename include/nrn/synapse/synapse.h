/// @file synapse.h
/// @brief Aggregate include for all built-in synapse models.
///
/// @details
/// Pulls in StaticSynapse, ExponentialSynapse, DoubleExponentialSynapse,
/// and AMPASynapse together with their option structs.
///
/// @see nrn/synapse/static.h, nrn/synapse/exponential.h,
///      nrn/synapse/double_exponential.h, nrn/synapse/ampa.h

#pragma once

#include <nrn/synapse/options.h>
#include <nrn/synapse/static.h>
#include <nrn/synapse/exponential.h>
#include <nrn/synapse/double_exponential.h>
#include <nrn/synapse/ampa.h>
