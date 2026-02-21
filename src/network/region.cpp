#include <nrn/network/region.h>

namespace nrn {

Region* region_create(const std::string& name) {
    auto* r = new Region();
    r->name = name;
    return r;
}

void region_destroy(Region* r) {
    delete r;
}

void region_add_population(Region* r, std::shared_ptr<Population> pop) {
    TORCH_CHECK(pop != nullptr,
                "Cannot add a null population to region '", r->name, "'");
    r->populations.push_back(std::move(pop));
}

void region_add_connection(Region* r, std::shared_ptr<Connection> conn) {
    TORCH_CHECK(conn != nullptr,
                "Cannot add a null connection to region '", r->name, "'");
    r->connections.push_back(std::move(conn));
}

std::shared_ptr<Population> region_find_population(const Region* r,
                                                   const std::string& name) {
    for (const auto& pop : r->populations) {
        if (pop->name == name) {
            return pop;
        }
    }
    return nullptr;
}

int64_t region_total_size(const Region* r) {
    int64_t total = 0;
    for (const auto& pop : r->populations) {
        total += pop->n;
    }
    return total;
}

} // namespace nrn
