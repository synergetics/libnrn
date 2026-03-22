/// @file connectivity.h
/// @brief Aggregate include for the connectivity subsystem.
///
/// @details
/// Pulls in ConnectivityTensor, ConnectOptions, the connect() free function,
/// and all built-in topology generators (Random, DistanceDependent).
///
/// @see ConnectivityTensor, connect(), TopologyGenerator

#pragma once

#include <nrn/connectivity/connectivity_tensor.h>
#include <nrn/connectivity/options.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/topology.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/connectivity/topology/distance_dependent.h>
