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

/// Create a new empty region with the given name.
inline Region* region_create(const std::string& name) {
    auto* r = new Region();
    r->name = name;
    return r;
}

/// Destroy a region and free its memory.
inline void region_destroy(Region* r) {
    delete r;
}

/// Add a population to the region.
inline void region_add_population(Region* r, std::shared_ptr<Population> pop) {
    TORCH_CHECK(pop != nullptr,
                "Cannot add a null population to region '", r->name, "'");
    r->populations.push_back(std::move(pop));
}

/// Add a connection to the region.
inline void region_add_connection(Region* r, std::shared_ptr<Connection> conn) {
    TORCH_CHECK(conn != nullptr,
                "Cannot add a null connection to region '", r->name, "'");
    r->connections.push_back(std::move(conn));
}

/// Find a population by name within the region. Returns nullptr if not found.
inline std::shared_ptr<Population> region_find_population(const Region* r,
                                                   const std::string& name) {
    for (const auto& pop : r->populations) {
        if (pop->name == name) {
            return pop;
        }
    }
    return nullptr;
}

/// Get the total number of neurons across all populations in the region.
inline int64_t region_total_size(const Region* r) {
    int64_t total = 0;
    for (const auto& pop : r->populations) {
        total += pop->n;
    }
    return total;
}

/// Get a string representation of the region's contents.
inline std::string region_repr(const Region* r) {
    std::string s = "Region '" + r->name + "':\n";
    for (const auto& pop : r->populations) {
        s += "  Population '" + pop->name + "' (n=" + std::to_string(pop->n) + ")\n";
    }
    s += "  Connections: " + std::to_string(r->connections.size()) + "\n";
    return s;
}

} // namespace nrn
