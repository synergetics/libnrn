#include <gtest/gtest.h>

#include <memory>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/neuron/lif.h>
#include <nrn/synapse/static.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/options.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/plasticity/stdp.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>
#include <nrn/simulation/options.h>

using namespace nrn;
using namespace nrn::literals;

// All GPU tests skip if CUDA is unavailable.
#define SKIP_IF_NO_CUDA                                       \
    do {                                                      \
        if (!torch::cuda::is_available()) {                   \
            GTEST_SKIP() << "CUDA not available, skipping.";  \
        }                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Device propagation: verify tensors land on the correct device.
// ---------------------------------------------------------------------------

TEST(GPUPipeline, DevicePropagation) {
    SKIP_IF_NO_CUDA;

    auto* lif = neuron::lif_create(100, neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).tau_m(20.0_ms)
        .i_bg(400.0e-12));
    auto pop = std::shared_ptr<Population>(
        population_create("neurons", neuron::lif_as_module(lif), 100, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.2));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn = connect(pop, pop, &gen, syn_mod,
                        ConnectOptions().block_size(32).weight_scale(500.0e-12));

    auto* region = region_create("gpu_test");
    region_add_population(region, pop);
    region_add_connection(region, conn);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.1_ms).duration(0.001).device(torch::kCUDA);

    auto* sim = sim_create(region, sim_opts);

    // After sim_create, population state tensors should be on CUDA.
    EXPECT_TRUE(state_get(pop->state, "v").is_cuda());
    EXPECT_TRUE(state_get(pop->state, "spike").is_cuda());
    EXPECT_TRUE(state_get(pop->state, "I_syn").is_cuda());

    // Connectivity tensors should be on CUDA.
    EXPECT_TRUE(conn->connectivity.weights.is_cuda());
    EXPECT_TRUE(conn->connectivity.structural_mask.is_cuda());

    sim_destroy(sim);
    region_destroy(region);
    conn.reset();
    pop.reset();
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(lif);
}

// ---------------------------------------------------------------------------
// Basic LIF run on GPU.
// ---------------------------------------------------------------------------

TEST(GPUPipeline, LIF_BasicRun) {
    SKIP_IF_NO_CUDA;

    const int64_t N = 1000;
    auto* lif = neuron::lif_create(N, neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(20.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(400.0e-12));
    auto pop = std::shared_ptr<Population>(
        population_create("neurons", neuron::lif_as_module(lif), N, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.1));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn = connect(pop, pop, &gen, syn_mod,
                        ConnectOptions().block_size(64).weight_scale(500.0e-12));

    auto* region = region_create("gpu_lif");
    region_add_population(region, pop);
    region_add_connection(region, conn);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.1_ms).dt_slow(10.0_ms).duration(0.05)
        .device(torch::kCUDA);

    auto* sim = sim_create(region, sim_opts);
    sim_record(sim, pop, {"spike"});

    // Run 500 steps (50 ms — enough time for spikes with i_bg ≈ 2× threshold).
    sim_run(sim);

    auto spikes = sim_get_spikes(sim, pop);
    // With suprathreshold background current, we expect spikes.
    EXPECT_GT(spikes.size(0), 0);

    sim_destroy(sim);
    region_destroy(region);
    conn.reset();
    pop.reset();
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(lif);
}

// ---------------------------------------------------------------------------
// E-I network on GPU.
// ---------------------------------------------------------------------------

TEST(GPUPipeline, EI_Network) {
    SKIP_IF_NO_CUDA;

    const int64_t N_exc = 400, N_inh = 100;
    double J_E = 500.0e-12, g = 5.0, J_I = -g * J_E;
    double i_bg = 380.0e-12;

    auto* exc_lif = neuron::lif_create(N_exc, neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(20.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(i_bg));
    auto* inh_lif = neuron::lif_create(N_inh, neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(10.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(i_bg));

    auto exc = std::shared_ptr<Population>(
        population_create("exc", neuron::lif_as_module(exc_lif), N_exc, torch::kCPU),
        population_destroy);
    auto inh = std::shared_ptr<Population>(
        population_create("inh", neuron::lif_as_module(inh_lif), N_inh, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.1));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto co_e = ConnectOptions().block_size(64).weight_scale(J_E);
    auto co_i = ConnectOptions().block_size(64).weight_scale(J_I);

    auto ee = connect(exc, exc, &gen, syn_mod, co_e);
    auto ei = connect(exc, inh, &gen, syn_mod, co_e);
    auto ie = connect(inh, exc, &gen, syn_mod, co_i);
    auto ii = connect(inh, inh, &gen, syn_mod, co_i);

    auto* region = region_create("gpu_ei");
    region_add_population(region, exc);
    region_add_population(region, inh);
    region_add_connection(region, ee);
    region_add_connection(region, ei);
    region_add_connection(region, ie);
    region_add_connection(region, ii);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.1_ms).dt_slow(10.0_ms).duration(0.05)
        .device(torch::kCUDA);

    auto* sim = sim_create(region, sim_opts);
    sim_record(sim, exc, {"spike"});
    sim_record(sim, inh, {"spike"});
    sim_run(sim);

    auto exc_spikes = sim_get_spikes(sim, exc);
    auto inh_spikes = sim_get_spikes(sim, inh);

    double exc_rate = static_cast<double>(exc_spikes.size(0)) / (N_exc * 0.05);
    double inh_rate = static_cast<double>(inh_spikes.size(0)) / (N_inh * 0.05);

    std::cout << "GPUPipeline_EI: exc_rate=" << exc_rate
              << " Hz, inh_rate=" << inh_rate << " Hz\n";

    EXPECT_GT(exc_spikes.size(0), 0);
    EXPECT_GT(inh_spikes.size(0), 0);
    EXPECT_LT(exc_rate, 200.0);  // not exploding
    EXPECT_LT(inh_rate, 200.0);

    sim_destroy(sim);
    region_destroy(region);
    ee.reset(); ei.reset(); ie.reset(); ii.reset();
    exc.reset(); inh.reset();
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
}

// ---------------------------------------------------------------------------
// STDP on GPU.
// ---------------------------------------------------------------------------

TEST(GPUPipeline, STDP_OnGPU) {
    SKIP_IF_NO_CUDA;

    // Use an E-I network so that recurrent activity desynchronizes neurons,
    // increasing the chance that STDP slow boundaries catch some spikes.
    const int64_t N_exc = 200, N_inh = 50;
    double J_E = 500.0e-12, g = 5.0, J_I = -g * J_E;
    double i_bg = 380.0e-12;

    auto* exc_lif = neuron::lif_create(N_exc, neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(20.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(i_bg));
    auto* inh_lif = neuron::lif_create(N_inh, neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(10.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(i_bg));

    auto exc = std::shared_ptr<Population>(
        population_create("exc", neuron::lif_as_module(exc_lif), N_exc, torch::kCPU),
        population_destroy);
    auto inh = std::shared_ptr<Population>(
        population_create("inh", neuron::lif_as_module(inh_lif), N_inh, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.2));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto co_e = ConnectOptions().block_size(32).weight_scale(J_E);
    auto co_i = ConnectOptions().block_size(32).weight_scale(J_I);

    auto ee = connect(exc, exc, &gen, syn_mod, co_e);
    auto ei = connect(exc, inh, &gen, syn_mod, co_e);
    auto ie = connect(inh, exc, &gen, syn_mod, co_i);
    auto ii = connect(inh, inh, &gen, syn_mod, co_i);

    auto initial_weights = ee->connectivity.weights.clone();

    auto* stdp = stdp_create(STDPOptions()
        .tau_plus(20.0_ms).tau_minus(20.0_ms)
        .a_plus(0.1).a_minus(-0.12)
        .learning_rate(1.0)
        .w_min(0.0).w_max(1.0));
    connection_attach(ee.get(), stdp_as_rule(stdp));

    auto* region = region_create("stdp_gpu");
    region_add_population(region, exc);
    region_add_population(region, inh);
    region_add_connection(region, ee);
    region_add_connection(region, ei);
    region_add_connection(region, ie);
    region_add_connection(region, ii);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.1_ms).dt_slow(1.0_ms).duration(0.1)
        .device(torch::kCUDA);

    auto* sim = sim_create(region, sim_opts);
    sim_record(sim, exc, {"spike"});
    sim_run(sim);

    auto spikes_rec = sim_get_spikes(sim, exc);
    std::cout << "GPUPipeline_STDP: spike count = " << spikes_rec.size(0) << "\n";

    // Weights should have changed.
    auto current_weights = ee->connectivity.weights.to(torch::kCPU);
    auto weight_diff = (current_weights - initial_weights).abs().sum().item<float>();
    std::cout << "GPUPipeline_STDP: total |dw| = " << weight_diff << "\n";
    EXPECT_GT(spikes_rec.size(0), 0);
    EXPECT_GT(weight_diff, 0.0f);

    sim_destroy(sim);
    region_destroy(region);
    ee.reset(); ei.reset(); ie.reset(); ii.reset();
    exc.reset(); inh.reset();
    stdp_destroy(stdp);
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
}
