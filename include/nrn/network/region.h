#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nrn/network/population.h>

namespace nrn {

// Forward-declare Connection.
class Connection;

/// A cortical region or brain area containing populations and connections.
/// Plain struct — just a named container.
struct Region {
    std::string name;
    std::vector<std::shared_ptr<Population>> populations;
    std::vector<std::shared_ptr<Connection>> connections;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Create an empty region.
Region* region_create(const std::string& name);

/// Destroy a region. Does NOT destroy populations/connections (shared_ptr).
void region_destroy(Region* r);

/// Add a population to the region.
void region_add_population(Region* r, std::shared_ptr<Population> pop);

/// Add a connection to the region.
void region_add_connection(Region* r, std::shared_ptr<Connection> conn);

/// Find a population by name. Returns nullptr if not found.
std::shared_ptr<Population> region_find_population(const Region* r,
                                                   const std::string& name);

/// Total number of neurons across all populations.
int64_t region_total_size(const Region* r);

} // namespace nrn
