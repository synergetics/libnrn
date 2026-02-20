#include <nrn/network/population.h>

#include <utility>

namespace nrn {

Population::Population(std::string name,
                       std::shared_ptr<torch::nn::Module> module,
                       int64_t n,
                       torch::Device device)
    : name_(std::move(name)),
      module_(std::move(module)),
      state_(),
      n_(n),
      positions_(std::nullopt) {
    TORCH_CHECK(n > 0, "Population '", name_, "' must have at least 1 neuron");
    // Move state tensors to the target device (the module itself should
    // already reside there — we leave that to the caller).
    (void)device; // State is empty at construction; tensors added later.
}

void Population::set_positions(torch::Tensor positions) {
    TORCH_CHECK(positions.dim() == 2 && positions.size(0) == n_ &&
                    positions.size(1) == 3,
                "positions must have shape [", n_, ", 3], got ",
                positions.sizes());
    positions_ = std::move(positions);
}

void Population::to(torch::Device device) {
    // Move every tensor in the state bag.
    for (auto& [key, tensor] : state_) {
        tensor = tensor.to(device);
    }
    // Move positions if present.
    if (positions_.has_value()) {
        positions_ = positions_->to(device);
    }
}

} // namespace nrn
