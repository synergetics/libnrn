/// @file region.h
/// @brief Region struct — a named container of Populations and Connections.
///
/// @details
/// A Region represents a cortical area or brain region.  It is the
/// top-level input to @c graph_compile(), which compiles it into an
/// execution @c Graph.
///
/// Populations and Connections are owned as @c shared_ptr; the compiled
/// Graph holds non-owning @c State* pointers into Population objects, so
/// **the Region must outlive any Graph compiled from it**.
///
/// @see Population, Connection, graph_compile()

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nrn/network/population.h>

namespace nrn {

// Forward-declare Connection.
class Connection;

/// @brief A cortical region or brain area containing populations and connections.
///
/// @details
/// Plain struct — just a named container.  Use the @c region_* free functions
/// to build and query the region.
struct Region {
    std::string name;                                    ///< Human-readable region name.
    std::vector<std::shared_ptr<Population>> populations; ///< All neuron populations in this region.
    std::vector<std::shared_ptr<Connection>> connections; ///< All synaptic connections in this region.
};

/// @brief Allocate an empty Region with the given name.
/// @param name  Human-readable region identifier.
/// @return Heap-allocated Region; caller takes ownership.
inline Region* region_create(const std::string& name) {
    auto* r = new Region();
    r->name = name;
    return r;
}

/// @brief Free a Region and release its shared Population/Connection references.
/// @param r  Pointer to the Region to destroy.
inline void region_destroy(Region* r) {
    delete r;
}

/// @brief Add a Population to the region.
/// @param r    Region to modify.
/// @param pop  Population to add; must not be null.
inline void region_add_population(Region* r, std::shared_ptr<Population> pop) {
    TORCH_CHECK(pop != nullptr,
                "Cannot add a null population to region '", r->name, "'");
    r->populations.push_back(std::move(pop));
}

/// @brief Add a Connection to the region.
/// @param r     Region to modify.
/// @param conn  Connection to add; must not be null.
inline void region_add_connection(Region* r, std::shared_ptr<Connection> conn) {
    TORCH_CHECK(conn != nullptr,
                "Cannot add a null connection to region '", r->name, "'");
    r->connections.push_back(std::move(conn));
}

/// @brief Find a population by name.
/// @param r     Region to search.
/// @param name  Population name to find.
/// @return Shared pointer to the matching Population, or @c nullptr if not found.
inline std::shared_ptr<Population> region_find_population(const Region* r,
                                                   const std::string& name) {
    for (const auto& pop : r->populations) {
        if (pop->name == name) {
            return pop;
        }
    }
    return nullptr;
}

/// @brief Return the total neuron count across all populations in the region.
/// @param r  Region to query.
/// @return Sum of @c pop->n for all populations.
inline int64_t region_total_size(const Region* r) {
    int64_t total = 0;
    for (const auto& pop : r->populations) {
        total += pop->n;
    }
    return total;
}

/// @brief Return a human-readable summary of the region for debugging.
/// @param r  Region to describe.
/// @return Multi-line string listing populations and connection count.
inline std::string region_repr(const Region* r) {
    std::string s = "Region '" + r->name + "':\n";
    for (const auto& pop : r->populations) {
        s += "  Population '" + pop->name + "' (n=" + std::to_string(pop->n) + ")\n";
    }
    s += "  Connections: " + std::to_string(r->connections.size()) + "\n";
    return s;
}

} // namespace nrn
