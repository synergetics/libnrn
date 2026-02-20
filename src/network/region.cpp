#include <nrn/network/region.h>

#include <stdexcept>

namespace nrn {

RegionImpl::RegionImpl(std::string region_name)
    : region_name_(std::move(region_name)) {
    // n_ (inherited from Module) starts at 0; updated as populations are added.
}

// ------------------------------------------------------------------
// Population / connection management
// ------------------------------------------------------------------

void RegionImpl::add(std::shared_ptr<Population> population) {
    TORCH_CHECK(population != nullptr,
                "Cannot add a null population to region '", region_name_, "'");
    n_ += population->size(); // keep Module::size() in sync
    populations_.push_back(std::move(population));
}

void RegionImpl::add(std::shared_ptr<Connection> connection) {
    TORCH_CHECK(connection != nullptr,
                "Cannot add a null connection to region '", region_name_, "'");
    connections_.push_back(std::move(connection));
}

// ------------------------------------------------------------------
// Queries
// ------------------------------------------------------------------

std::shared_ptr<Population> RegionImpl::population(
    const std::string& name) const {
    for (const auto& pop : populations_) {
        if (pop->name() == name) {
            return pop;
        }
    }
    TORCH_CHECK(false, "Region '", region_name_,
                "' does not contain population '", name, "'");
    return nullptr; // unreachable
}

int64_t RegionImpl::total_size() const {
    int64_t total = 0;
    for (const auto& pop : populations_) {
        total += pop->size();
    }
    return total;
}

// ------------------------------------------------------------------
// Module interface
// ------------------------------------------------------------------

void RegionImpl::forward(State& /*state*/, Time /*t*/, Duration /*dt*/) {
    TORCH_CHECK(false,
                "Region::forward() not yet implemented — use Simulation to "
                "orchestrate the time-stepping loop");
}

void RegionImpl::reset() {
    // Reset every population's state.
    for (auto& pop : populations_) {
        // Populations are not Modules themselves, so we clear their state.
        pop->state() = State();
    }
    // Connections will be reset when they gain a Module interface.
}

std::vector<std::string> RegionImpl::state_vars() const {
    // Aggregate from all populations (prefixed by population name).
    std::vector<std::string> vars;
    for (const auto& pop : populations_) {
        for (const auto& key : pop->state().keys()) {
            vars.push_back(pop->name() + "/" + key);
        }
    }
    return vars;
}

} // namespace nrn
