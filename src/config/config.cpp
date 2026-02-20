#include <nrn/config/config.h>

#include <stdexcept>

namespace nrn {
namespace config {

Simulation load(const std::string& /*path*/) {
    throw std::runtime_error(
        "nrn::config::load() not yet implemented — YAML config loading "
        "is planned for Phase 1");
}

Region load_region(const std::string& /*path*/) {
    throw std::runtime_error(
        "nrn::config::load_region() not yet implemented — YAML config "
        "loading is planned for Phase 1");
}

} // namespace config
} // namespace nrn
