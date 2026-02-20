#pragma once

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/module.h>
#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/network/population.h>

namespace nrn {

// Forward-declare Connection (defined in connectivity/connect.h) so that
// region.h does not force a dependency on the full connectivity header tree.
class Connection;

/// A cortical region or brain area containing populations and connections.
///
/// RegionImpl is a Module that composes populations of neurons (and
/// eventually glia) with their inter-population connections.  It owns the
/// populations and connections via shared pointers so that external code
/// can retain handles.
///
/// Use via the TORCH_MODULE wrapper: `nrn::Region`.
class RegionImpl : public nrn::Module<RegionImpl> {
public:
    /// Construct an empty region with the given name.
    explicit RegionImpl(std::string region_name = "region");

    // ------------------------------------------------------------------
    // Population management
    // ------------------------------------------------------------------

    /// Add a single population.
    void add(std::shared_ptr<Population> population);

    /// Add a single connection.
    void add(std::shared_ptr<Connection> connection);

    /// Variadic helper — add multiple populations / connections in one call.
    template <typename First, typename... Rest>
    void add(First&& first, Rest&&... rest) {
        add(std::forward<First>(first));
        add(std::forward<Rest>(rest)...);
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Find a population by name.  Throws if not found.
    std::shared_ptr<Population> population(const std::string& name) const;

    /// All populations.
    const std::vector<std::shared_ptr<Population>>& populations() const {
        return populations_;
    }

    /// All connections.
    const std::vector<std::shared_ptr<Connection>>& connections() const {
        return connections_;
    }

    /// Total number of neurons across all populations.
    int64_t total_size() const;

    // ------------------------------------------------------------------
    // Module interface
    // ------------------------------------------------------------------

    /// Advance all populations and connections by one timestep.
    void forward(State& state, Time t, Duration dt) override;

    /// Re-initialize parameters and state for all children.
    void reset() override;

    /// State variable names (aggregated from all children).
    std::vector<std::string> state_vars() const override;

private:
    std::string region_name_;
    std::vector<std::shared_ptr<Population>> populations_;
    std::vector<std::shared_ptr<Connection>> connections_;
};

TORCH_MODULE(Region);

} // namespace nrn
