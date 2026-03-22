/// @file config.h
/// @brief YAML-based configuration loader for Simulations and Regions.
///
/// @details
/// Provides two factory functions that deserialise a YAML configuration
/// file into fully constructed Simulation or Region objects.  The concrete
/// implementation is still a stub (throws @c std::runtime_error); these
/// declarations define the intended public API.
///
/// @see Simulation, Region

#pragma once

#include <string>

#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>

namespace nrn {
namespace config {

/// @brief Load a complete Simulation from a YAML configuration file.
///
/// @param path  Path to the YAML configuration file.
/// @return Heap-allocated Simulation; caller must call @c sim_destroy().
///
/// @throws std::runtime_error  Currently always — implementation pending.
///
/// @see sim_create(), sim_destroy()
Simulation* load(const std::string& path);

/// @brief Load a Region from a YAML configuration file.
///
/// @param path  Path to the YAML configuration file.
/// @return Heap-allocated Region; caller must call @c region_destroy().
///
/// @throws std::runtime_error  Currently always — implementation pending.
///
/// @see region_create(), region_destroy()
Region* load_region(const std::string& path);

} // namespace config
} // namespace nrn
