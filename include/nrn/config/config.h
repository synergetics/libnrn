#pragma once

#include <string>

#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>

namespace nrn {
namespace config {

/// Load a complete Simulation from a YAML configuration file.
///
/// The config file specifies populations, connections, plasticity rules,
/// astrocytes, neuromodulation, and simulation parameters.  Returns a
/// fully-constructed Simulation ready to call `run()`.
///
/// @throws std::runtime_error  Currently always — not yet implemented.
Simulation load(const std::string& path);

/// Load a Region (populations + connections) from a YAML file.
///
/// @throws std::runtime_error  Currently always — not yet implemented.
Region load_region(const std::string& path);

} // namespace config
} // namespace nrn
