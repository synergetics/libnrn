#include <nrn/plasticity/stdp.h>
#include <nrn/connectivity/connectivity_tensor.h>

#include <cmath>

namespace nrn {

STDP::STDP(const STDPOptions& opts)
    : opts_(opts) {}

void STDP::initialize(ConnectivityTensor& conn) {
    TORCH_CHECK(conn.weights.defined(),
                "STDP::initialize: weights tensor must be defined");

    // Allocate trace tensors with the same shape and device as weights,
    // initialized to zero.
    conn.trace_pre  = torch::zeros_like(conn.weights);
    conn.trace_post = torch::zeros_like(conn.weights);
}

void STDP::update(ConnectivityTensor& /*conn*/,
                  const State& /*pre_state*/,
                  const State& /*post_state*/,
                  Time /*t*/,
                  Duration dt) {
    // Cache decay factors when dt changes.
    if (dt != cached_dt_) {
        cached_dt_  = dt;
        decay_pre_  = std::exp(-dt / opts_.tau_plus());
        decay_post_ = std::exp(-dt / opts_.tau_minus());
    }

    TORCH_CHECK(false,
                "STDP::update() not yet implemented — "
                "requires spike tensor extraction from pre/post State "
                "and element-wise trace decay + weight update kernels");
}

void STDP::reset() {
    // Traces live inside the ConnectivityTensor, not here.
    // A full reset requires access to the ConnectivityTensor:
    //   conn.trace_pre->zero_();
    //   conn.trace_post->zero_();
    // This method resets only local cached state.
    cached_dt_  = 0.0;
    decay_pre_  = 0.0;
    decay_post_ = 0.0;
}

} // namespace nrn
