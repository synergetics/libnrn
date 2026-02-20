#include <gtest/gtest.h>

#include <memory>

#include <torch/torch.h>

// Include the umbrella header to verify it compiles cleanly.
#include <nrn.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/neuron/lif.h>
#include <nrn/neuron/adex.h>
#include <nrn/neuron/izhikevich.h>
#include <nrn/synapse/static.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/options.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/plasticity/stdp.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>
#include <nrn/simulation/options.h>
#include <nrn/simulation/clock.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Integration placeholder: verify that the umbrella header compiles and
// all major types are accessible.
// ---------------------------------------------------------------------------

TEST(Integration, Placeholder) {
    SUCCEED();
}

TEST(Integration, UmbrellaHeaderCompiles) {
    // Verify that key types from each subsystem are visible.
    nrn::Time t = 0.0;
    nrn::Duration dt = 0.0001;
    (void)t;
    (void)dt;

    nrn::State state;
    EXPECT_EQ(state.size(), 0u);
}

// ---------------------------------------------------------------------------
// LIF population construction
// ---------------------------------------------------------------------------

TEST(Integration, CreateLIFPopulations) {
    auto exc_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV)
        .v_thresh(-50.0_mV)
        .v_reset(-65.0_mV)
        .tau_m(20.0_ms)
        .tau_ref(2.0_ms)
        .c_m(250.0_pF);

    auto inh_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV)
        .v_thresh(-50.0_mV)
        .v_reset(-65.0_mV)
        .tau_m(10.0_ms)
        .tau_ref(2.0_ms)
        .c_m(250.0_pF);

    neuron::LIF exc_module(8000, exc_opts);
    neuron::LIF inh_module(2000, inh_opts);

    EXPECT_EQ(exc_module->size(), 8000);
    EXPECT_EQ(inh_module->size(), 2000);
}

// ---------------------------------------------------------------------------
// Population wrapper construction
// ---------------------------------------------------------------------------

TEST(Integration, PopulationConstruction) {
    auto lif = neuron::LIF(100);

    auto pop = std::make_shared<Population>(
        "test_pop",
        lif.ptr(),
        100,
        torch::kCPU);

    EXPECT_EQ(pop->name(), "test_pop");
    EXPECT_EQ(pop->size(), 100);
}

TEST(Integration, MultiplePopulations) {
    auto exc = neuron::LIF(1000);
    auto inh = neuron::LIF(250);

    auto pop_exc = std::make_shared<Population>(
        "excitatory",
        exc.ptr(),
        1000,
        torch::kCPU);

    auto pop_inh = std::make_shared<Population>(
        "inhibitory",
        inh.ptr(),
        250,
        torch::kCPU);

    EXPECT_EQ(pop_exc->size(), 1000);
    EXPECT_EQ(pop_inh->size(), 250);
    EXPECT_NE(pop_exc->name(), pop_inh->name());
}

// ---------------------------------------------------------------------------
// Random connectivity between populations
// ---------------------------------------------------------------------------

TEST(Integration, RandomConnectivity) {
    auto exc = neuron::LIF(64);
    auto pop_exc = std::make_shared<Population>(
        "exc", exc.ptr(), 64, torch::kCPU);

    auto random_topo = Random(RandomTopologyOptions().probability(0.1));

    auto syn = StaticSynapse(1);
    auto conn_opts = ConnectOptions()
        .block_size(32)
        .representation(BlockDense);

    auto conn = connect(pop_exc, pop_exc, random_topo,
                        syn.ptr(), conn_opts);

    EXPECT_EQ(conn->connectivity().n_source, 64);
    EXPECT_EQ(conn->connectivity().n_target, 64);
    EXPECT_EQ(conn->connectivity().block_size, 32);
}

// ---------------------------------------------------------------------------
// STDP attachment to connection
// ---------------------------------------------------------------------------

TEST(Integration, STDPAttachment) {
    auto lif = neuron::LIF(32);
    auto pop = std::make_shared<Population>(
        "neurons", lif.ptr(), 32, torch::kCPU);

    auto topo = Random(RandomTopologyOptions().probability(0.2));
    auto syn = StaticSynapse(1);
    auto conn = connect(pop, pop, topo, syn.ptr(),
                        ConnectOptions().block_size(16));

    auto stdp = std::make_shared<STDP>(STDPOptions()
        .tau_plus(20.0_ms)
        .tau_minus(20.0_ms)
        .a_plus(0.01)
        .a_minus(-0.012));

    // Attaching STDP should not crash.
    conn->attach(stdp);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Region composition
// ---------------------------------------------------------------------------

TEST(Integration, RegionComposition) {
    auto exc_lif = neuron::LIF(800);
    auto inh_lif = neuron::LIF(200);

    auto exc = std::make_shared<Population>(
        "excitatory", exc_lif.ptr(), 800, torch::kCPU);
    auto inh = std::make_shared<Population>(
        "inhibitory", inh_lif.ptr(), 200, torch::kCPU);

    Region region("test_region");
    region->add(exc, inh);

    EXPECT_EQ(region->populations().size(), 2u);
    EXPECT_EQ(region->total_size(), 1000);
}

TEST(Integration, RegionPopulationLookup) {
    auto lif = neuron::LIF(100);
    auto pop = std::make_shared<Population>(
        "my_pop", lif.ptr(), 100, torch::kCPU);

    Region region("lookup_test");
    region->add(pop);

    auto found = region->population("my_pop");
    EXPECT_EQ(found->name(), "my_pop");
    EXPECT_EQ(found->size(), 100);
}

TEST(Integration, RegionWithConnections) {
    auto exc_lif = neuron::LIF(64);
    auto inh_lif = neuron::LIF(16);

    auto exc = std::make_shared<Population>(
        "exc", exc_lif.ptr(), 64, torch::kCPU);
    auto inh = std::make_shared<Population>(
        "inh", inh_lif.ptr(), 16, torch::kCPU);

    auto topo = Random(RandomTopologyOptions().probability(0.1));
    auto syn = StaticSynapse(1);
    auto conn_opts = ConnectOptions().block_size(16);

    auto ee = connect(exc, exc, topo, syn.ptr(), conn_opts);
    auto ei = connect(exc, inh, topo, syn.ptr(), conn_opts);
    auto ie = connect(inh, exc, topo, syn.ptr(), conn_opts);
    auto ii = connect(inh, inh, topo, syn.ptr(), conn_opts);

    Region region("ei_network");
    region->add(exc, inh);
    region->add(ee, ei, ie, ii);

    EXPECT_EQ(region->populations().size(), 2u);
    EXPECT_EQ(region->connections().size(), 4u);
    EXPECT_EQ(region->total_size(), 80);
}

// ---------------------------------------------------------------------------
// Clock construction for simulation
// ---------------------------------------------------------------------------

TEST(Integration, SimulationClockTimescales) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    EXPECT_DOUBLE_EQ(clk.dt_fast(), 0.1e-3);
    EXPECT_DOUBLE_EQ(clk.dt_slow(), 10.0e-3);
    EXPECT_DOUBLE_EQ(clk.dt_structural(), 1.0);
}

// ---------------------------------------------------------------------------
// Simulation options
// ---------------------------------------------------------------------------

TEST(Integration, SimulationOptionsDefaults) {
    SimulationOptions opts;

    EXPECT_DOUBLE_EQ(opts.dt_fast(), 0.0001);
    EXPECT_DOUBLE_EQ(opts.dt_slow(), 0.01);
    EXPECT_DOUBLE_EQ(opts.dt_structural(), 1.0);
    EXPECT_DOUBLE_EQ(opts.duration(), 1.0);
}

TEST(Integration, SimulationOptionsCustom) {
    auto opts = SimulationOptions()
        .dt_fast(0.1_ms)
        .dt_slow(10.0_ms)
        .dt_structural(1.0_s)
        .duration(5.0_s)
        .device(torch::kCPU);

    EXPECT_DOUBLE_EQ(opts.dt_fast(), 0.1e-3);
    EXPECT_DOUBLE_EQ(opts.duration(), 5.0);
}

// ---------------------------------------------------------------------------
// Full Brunel-like network assembly (construction only, no run)
// ---------------------------------------------------------------------------

TEST(Integration, BrunelNetworkAssembly) {
    // This test assembles a complete E-I network similar to Brunel (2000),
    // verifying that all components can be constructed and composed without
    // crashing. It does NOT run the simulation (that requires completed
    // kernel implementations).

    const int64_t N_exc = 800;
    const int64_t N_inh = 200;
    const int64_t block_size = 64;

    // Neuron models.
    auto exc_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV)
        .v_thresh(-50.0_mV)
        .tau_m(20.0_ms);

    auto inh_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV)
        .v_thresh(-50.0_mV)
        .tau_m(10.0_ms);

    auto exc_module = neuron::LIF(N_exc, exc_opts);
    auto inh_module = neuron::LIF(N_inh, inh_opts);

    // Populations.
    auto exc = std::make_shared<Population>(
        "excitatory", exc_module.ptr(), N_exc, torch::kCPU);
    auto inh = std::make_shared<Population>(
        "inhibitory", inh_module.ptr(), N_inh, torch::kCPU);

    EXPECT_EQ(exc->size(), N_exc);
    EXPECT_EQ(inh->size(), N_inh);

    // Connectivity.
    auto topo = Random(RandomTopologyOptions().probability(0.1).allow_autapses(false));
    auto syn = StaticSynapse(1);
    auto conn_opts = ConnectOptions().block_size(block_size).representation(BlockDense);

    auto ee = connect(exc, exc, topo, syn.ptr(), conn_opts);
    auto ei = connect(exc, inh, topo, syn.ptr(), conn_opts);
    auto ie = connect(inh, exc, topo, syn.ptr(), conn_opts);
    auto ii = connect(inh, inh, topo, syn.ptr(), conn_opts);

    // Plasticity on E-E.
    auto stdp = std::make_shared<STDP>(STDPOptions()
        .tau_plus(20.0_ms)
        .tau_minus(20.0_ms)
        .a_plus(0.01)
        .a_minus(-0.012));
    ee->attach(stdp);

    // Region.
    Region region("brunel");
    region->add(exc, inh);
    region->add(ee, ei, ie, ii);

    EXPECT_EQ(region->populations().size(), 2u);
    EXPECT_EQ(region->connections().size(), 4u);
    EXPECT_EQ(region->total_size(), N_exc + N_inh);

    // Verify connectivity tensors have expected dimensions.
    EXPECT_EQ(ee->connectivity().n_source, N_exc);
    EXPECT_EQ(ee->connectivity().n_target, N_exc);
    EXPECT_EQ(ei->connectivity().n_source, N_exc);
    EXPECT_EQ(ei->connectivity().n_target, N_inh);
    EXPECT_EQ(ie->connectivity().n_source, N_inh);
    EXPECT_EQ(ie->connectivity().n_target, N_exc);
    EXPECT_EQ(ii->connectivity().n_source, N_inh);
    EXPECT_EQ(ii->connectivity().n_target, N_inh);

    // Verify structural masks exist.
    EXPECT_TRUE(ee->connectivity().structural_mask.defined());
    EXPECT_TRUE(ee->connectivity().weights.defined());
}

// ---------------------------------------------------------------------------
// Mixed neuron models in one region
// ---------------------------------------------------------------------------

TEST(Integration, MixedNeuronModels) {
    auto lif = neuron::LIF(100);
    auto adex = neuron::AdEx(100);
    auto izh = neuron::Izhikevich(100);

    auto pop_lif = std::make_shared<Population>(
        "lif_pop", lif.ptr(), 100, torch::kCPU);
    auto pop_adex = std::make_shared<Population>(
        "adex_pop", adex.ptr(), 100, torch::kCPU);
    auto pop_izh = std::make_shared<Population>(
        "izh_pop", izh.ptr(), 100, torch::kCPU);

    Region region("mixed_models");
    region->add(pop_lif, pop_adex, pop_izh);

    EXPECT_EQ(region->populations().size(), 3u);
    EXPECT_EQ(region->total_size(), 300);
}

// ---------------------------------------------------------------------------
// State bag for populations
// ---------------------------------------------------------------------------

TEST(Integration, PopulationState) {
    auto lif = neuron::LIF(50);
    auto pop = std::make_shared<Population>(
        "test", lif.ptr(), 50, torch::kCPU);

    // The state bag should be accessible.
    State& state = pop->state();
    EXPECT_EQ(state.size(), 0u);  // Empty until populated by simulation.

    // We can add tensors to the state.
    state.set("custom", torch::zeros({50}));
    EXPECT_TRUE(state.has("custom"));
    EXPECT_EQ(state.get("custom").size(0), 50);
}
