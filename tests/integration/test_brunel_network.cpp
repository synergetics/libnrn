#include <gtest/gtest.h>

#include <memory>

#include <torch/torch.h>

// Include the umbrella header to verify it compiles cleanly.
#include <nrn/nrn.h>

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

    auto* exc_lif = neuron::lif_create(8000, exc_opts);
    auto* inh_lif = neuron::lif_create(2000, inh_opts);

    EXPECT_EQ(exc_lif->n, 8000);
    EXPECT_EQ(inh_lif->n, 2000);

    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
}

// ---------------------------------------------------------------------------
// Population wrapper construction
// ---------------------------------------------------------------------------

TEST(Integration, PopulationConstruction) {
    auto* lif = neuron::lif_create(100);
    auto lif_mod = neuron::lif_as_module(lif);

    auto pop = std::shared_ptr<Population>(
        population_create("test_pop", lif_mod, 100, torch::kCPU),
        population_destroy);

    EXPECT_EQ(pop->name, "test_pop");
    EXPECT_EQ(pop->n, 100);

    // lif is owned externally; destroy after pop is done.
    pop.reset();
    neuron::lif_destroy(lif);
}

TEST(Integration, MultiplePopulations) {
    auto* exc_lif = neuron::lif_create(1000);
    auto* inh_lif = neuron::lif_create(250);

    auto pop_exc = std::shared_ptr<Population>(
        population_create("excitatory", neuron::lif_as_module(exc_lif), 1000, torch::kCPU),
        population_destroy);

    auto pop_inh = std::shared_ptr<Population>(
        population_create("inhibitory", neuron::lif_as_module(inh_lif), 250, torch::kCPU),
        population_destroy);

    EXPECT_EQ(pop_exc->n, 1000);
    EXPECT_EQ(pop_inh->n, 250);
    EXPECT_NE(pop_exc->name, pop_inh->name);

    pop_exc.reset();
    pop_inh.reset();
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
}

// ---------------------------------------------------------------------------
// Random connectivity between populations
// ---------------------------------------------------------------------------

TEST(Integration, RandomConnectivity) {
    auto* lif = neuron::lif_create(64);
    auto lif_mod = neuron::lif_as_module(lif);
    auto pop_exc = std::shared_ptr<Population>(
        population_create("exc", lif_mod, 64, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.1));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn_opts = ConnectOptions()
        .block_size(32)
        .representation(BlockDense);

    auto conn = connect(pop_exc, pop_exc, &gen, syn_mod, conn_opts);

    EXPECT_EQ(conn->connectivity.n_source, 64);
    EXPECT_EQ(conn->connectivity.n_target, 64);
    EXPECT_EQ(conn->connectivity.block_size, 32);

    conn.reset();
    pop_exc.reset();
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(lif);
}

// ---------------------------------------------------------------------------
// STDP attachment to connection
// ---------------------------------------------------------------------------

TEST(Integration, STDPAttachment) {
    auto* lif = neuron::lif_create(32);
    auto lif_mod = neuron::lif_as_module(lif);
    auto pop = std::shared_ptr<Population>(
        population_create("neurons", lif_mod, 32, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.2));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn = connect(pop, pop, &gen, syn_mod,
                        ConnectOptions().block_size(16));

    auto* stdp = stdp_create(STDPOptions()
        .tau_plus(20.0_ms)
        .tau_minus(20.0_ms)
        .a_plus(0.01)
        .a_minus(-0.012));

    // Attaching STDP should not crash.
    connection_attach(conn.get(), stdp_as_rule(stdp));
    SUCCEED();

    conn.reset();
    pop.reset();
    stdp_destroy(stdp);
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(lif);
}

// ---------------------------------------------------------------------------
// Region composition
// ---------------------------------------------------------------------------

TEST(Integration, RegionComposition) {
    auto* exc_lif = neuron::lif_create(800);
    auto* inh_lif = neuron::lif_create(200);

    auto exc = std::shared_ptr<Population>(
        population_create("excitatory", neuron::lif_as_module(exc_lif), 800, torch::kCPU),
        population_destroy);
    auto inh = std::shared_ptr<Population>(
        population_create("inhibitory", neuron::lif_as_module(inh_lif), 200, torch::kCPU),
        population_destroy);

    auto* region = region_create("test_region");
    region_add_population(region, exc);
    region_add_population(region, inh);

    EXPECT_EQ(region->populations.size(), 2u);
    EXPECT_EQ(region_total_size(region), 1000);

    region_destroy(region);
    exc.reset();
    inh.reset();
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
}

TEST(Integration, RegionPopulationLookup) {
    auto* lif = neuron::lif_create(100);
    auto lif_mod = neuron::lif_as_module(lif);
    auto pop = std::shared_ptr<Population>(
        population_create("my_pop", lif_mod, 100, torch::kCPU),
        population_destroy);

    auto* region = region_create("lookup_test");
    region_add_population(region, pop);

    auto found = region_find_population(region, "my_pop");
    EXPECT_EQ(found->name, "my_pop");
    EXPECT_EQ(found->n, 100);

    region_destroy(region);
    pop.reset();
    neuron::lif_destroy(lif);
}

TEST(Integration, RegionWithConnections) {
    auto* exc_lif = neuron::lif_create(64);
    auto* inh_lif = neuron::lif_create(16);

    auto exc = std::shared_ptr<Population>(
        population_create("exc", neuron::lif_as_module(exc_lif), 64, torch::kCPU),
        population_destroy);
    auto inh = std::shared_ptr<Population>(
        population_create("inh", neuron::lif_as_module(inh_lif), 16, torch::kCPU),
        population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.1));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn_opts = ConnectOptions().block_size(16);

    auto ee = connect(exc, exc, &gen, syn_mod, conn_opts);
    auto ei = connect(exc, inh, &gen, syn_mod, conn_opts);
    auto ie = connect(inh, exc, &gen, syn_mod, conn_opts);
    auto ii = connect(inh, inh, &gen, syn_mod, conn_opts);

    auto* region = region_create("ei_network");
    region_add_population(region, exc);
    region_add_population(region, inh);
    region_add_connection(region, ee);
    region_add_connection(region, ei);
    region_add_connection(region, ie);
    region_add_connection(region, ii);

    EXPECT_EQ(region->populations.size(), 2u);
    EXPECT_EQ(region->connections.size(), 4u);
    EXPECT_EQ(region_total_size(region), 80);

    region_destroy(region);
    ee.reset(); ei.reset(); ie.reset(); ii.reset();
    exc.reset(); inh.reset();
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
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

    auto* exc_lif = neuron::lif_create(N_exc, exc_opts);
    auto* inh_lif = neuron::lif_create(N_inh, inh_opts);

    // Populations.
    auto exc = std::shared_ptr<Population>(
        population_create("excitatory", neuron::lif_as_module(exc_lif), N_exc, torch::kCPU),
        population_destroy);
    auto inh = std::shared_ptr<Population>(
        population_create("inhibitory", neuron::lif_as_module(inh_lif), N_inh, torch::kCPU),
        population_destroy);

    EXPECT_EQ(exc->n, N_exc);
    EXPECT_EQ(inh->n, N_inh);

    // Connectivity.
    auto* topo = random_topology_create(
        RandomTopologyOptions().probability(0.1).allow_autapses(false));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn_opts = ConnectOptions().block_size(block_size).representation(BlockDense);

    auto ee = connect(exc, exc, &gen, syn_mod, conn_opts);
    auto ei = connect(exc, inh, &gen, syn_mod, conn_opts);
    auto ie = connect(inh, exc, &gen, syn_mod, conn_opts);
    auto ii = connect(inh, inh, &gen, syn_mod, conn_opts);

    // Plasticity on E-E.
    auto* stdp = stdp_create(STDPOptions()
        .tau_plus(20.0_ms)
        .tau_minus(20.0_ms)
        .a_plus(0.01)
        .a_minus(-0.012));
    connection_attach(ee.get(), stdp_as_rule(stdp));

    // Region.
    auto* region = region_create("brunel");
    region_add_population(region, exc);
    region_add_population(region, inh);
    region_add_connection(region, ee);
    region_add_connection(region, ei);
    region_add_connection(region, ie);
    region_add_connection(region, ii);

    EXPECT_EQ(region->populations.size(), 2u);
    EXPECT_EQ(region->connections.size(), 4u);
    EXPECT_EQ(region_total_size(region), N_exc + N_inh);

    // Verify connectivity tensors have expected dimensions.
    EXPECT_EQ(ee->connectivity.n_source, N_exc);
    EXPECT_EQ(ee->connectivity.n_target, N_exc);
    EXPECT_EQ(ei->connectivity.n_source, N_exc);
    EXPECT_EQ(ei->connectivity.n_target, N_inh);
    EXPECT_EQ(ie->connectivity.n_source, N_inh);
    EXPECT_EQ(ie->connectivity.n_target, N_exc);
    EXPECT_EQ(ii->connectivity.n_source, N_inh);
    EXPECT_EQ(ii->connectivity.n_target, N_inh);

    // Verify structural masks exist.
    EXPECT_TRUE(ee->connectivity.structural_mask.defined());
    EXPECT_TRUE(ee->connectivity.weights.defined());

    // Cleanup.
    region_destroy(region);
    ee.reset(); ei.reset(); ie.reset(); ii.reset();
    exc.reset(); inh.reset();
    stdp_destroy(stdp);
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);
}

// ---------------------------------------------------------------------------
// Mixed neuron models in one region
// ---------------------------------------------------------------------------

TEST(Integration, MixedNeuronModels) {
    auto* lif = neuron::lif_create(100);
    auto* adex = neuron::adex_create(100);
    auto* izh = neuron::izh_create(100);

    auto pop_lif = std::shared_ptr<Population>(
        population_create("lif_pop", neuron::lif_as_module(lif), 100, torch::kCPU),
        population_destroy);
    auto pop_adex = std::shared_ptr<Population>(
        population_create("adex_pop", neuron::adex_as_module(adex), 100, torch::kCPU),
        population_destroy);
    auto pop_izh = std::shared_ptr<Population>(
        population_create("izh_pop", neuron::izh_as_module(izh), 100, torch::kCPU),
        population_destroy);

    auto* region = region_create("mixed_models");
    region_add_population(region, pop_lif);
    region_add_population(region, pop_adex);
    region_add_population(region, pop_izh);

    EXPECT_EQ(region->populations.size(), 3u);
    EXPECT_EQ(region_total_size(region), 300);

    region_destroy(region);
    pop_lif.reset(); pop_adex.reset(); pop_izh.reset();
    neuron::lif_destroy(lif);
    neuron::adex_destroy(adex);
    neuron::izh_destroy(izh);
}

// ---------------------------------------------------------------------------
// State bag for populations
// ---------------------------------------------------------------------------

TEST(Integration, PopulationState) {
    auto* lif = neuron::lif_create(50);
    auto lif_mod = neuron::lif_as_module(lif);
    auto pop = std::shared_ptr<Population>(
        population_create("test", lif_mod, 50, torch::kCPU),
        population_destroy);

    // The state bag should be accessible.
    State& state = pop->state;
    EXPECT_EQ(state.size(), 0u);  // Empty until populated by simulation.

    // We can add tensors to the state.
    state_set(state, "custom", torch::zeros({50}));
    EXPECT_TRUE(state_contains(state, "custom"));
    EXPECT_EQ(state_get(state, "custom").size(0), 50);

    pop.reset();
    neuron::lif_destroy(lif);
}
