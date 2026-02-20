#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/topology.h>
#include <nrn/network/population.h>
#include <nrn/plasticity/plasticity_rule.h>

namespace nrn {

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

Connection::Connection(std::shared_ptr<Population> source,
                       std::shared_ptr<Population> target,
                       ConnectivityTensor connectivity,
                       std::shared_ptr<torch::nn::Module> synapse,
                       ConnectOptions options)
    : source_(std::move(source)),
      target_(std::move(target)),
      connectivity_(std::move(connectivity)),
      synapse_(std::move(synapse)),
      options_(std::move(options)) {}

void Connection::deliver(const torch::Tensor& /*spikes*/,
                         Time /*t*/, Duration /*dt*/) {
    TORCH_CHECK(false,
                "Connection::deliver() not yet implemented — "
                "requires spike buffer and CUDA kernel integration");
}

void Connection::attach(std::shared_ptr<PlasticityRule> rule) {
    rule->initialize(connectivity_);
    plasticity_rules_.push_back(std::move(rule));
}

void Connection::update_plasticity(const State& pre_state,
                                   const State& post_state,
                                   Time t, Duration dt) {
    for (auto& rule : plasticity_rules_) {
        rule->update(connectivity_, pre_state, post_state, t, dt);
    }
}

// ---------------------------------------------------------------------------
// Free function: connect()
// ---------------------------------------------------------------------------

std::shared_ptr<Connection> connect(
    std::shared_ptr<Population> source,
    std::shared_ptr<Population> target,
    TopologyGenerator& topology,
    std::shared_ptr<torch::nn::Module> synapse,
    const ConnectOptions& options) {
    // Use the topology generator to create the block structure and masks.
    // For now we pass CPU as default device; the caller can .to(device) later.
    auto ct = topology.generate(
        source->size(),
        target->size(),
        options.block_size(),
        torch::kCPU);

    return std::make_shared<Connection>(
        std::move(source),
        std::move(target),
        std::move(ct),
        std::move(synapse),
        options);
}

} // namespace nrn
