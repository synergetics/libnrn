#include <nrn/network/population.h>

namespace nrn {

Population* population_create(const std::string& name,
                              NrnModule module,
                              int64_t n,
                              torch::Device /*device*/) {
    TORCH_CHECK(n > 0, "Population '", name, "' must have at least 1 neuron");
    auto* pop = new Population();
    pop->name = name;
    pop->module = module;
    pop->n = n;
    pop->positions = std::nullopt;
    // State is empty at construction; populated by first forward() call.
    return pop;
}

void population_destroy(Population* pop) {
    delete pop;
}

void population_set_positions(Population* pop, torch::Tensor positions) {
    TORCH_CHECK(positions.dim() == 2 && positions.size(0) == pop->n &&
                    positions.size(1) == 3,
                "positions must have shape [", pop->n, ", 3], got ",
                positions.sizes());
    pop->positions = std::move(positions);
}

void population_to_device(Population* pop, torch::Device device) {
    state_to_device(pop->state, device);
    if (pop->positions.has_value()) {
        pop->positions = pop->positions->to(device);
    }
    nrn_to_device(&pop->module, device);
}

} // namespace nrn
