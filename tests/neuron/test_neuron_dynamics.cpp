#include <gtest/gtest.h>
#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/neuron/lif.h>
#include <nrn/neuron/adex.h>
#include <nrn/neuron/izhikevich.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/synapse/static.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>
#include <nrn/simulation/options.h>

using namespace nrn;
using namespace nrn::literals;

// ===========================================================================
// LIF Dynamics
// ===========================================================================

TEST(LIFDynamics, ConstantCurrentProducesSpikes) {
    // V_inf = v_rest + I * tau_m / c_m = -0.065 + 500e-12 * 0.01 / 250e-12
    //       = -0.065 + 0.02 = -0.045 V (above threshold -0.050)
    auto opts = neuron::LIFOptions()
        .v_rest(-0.065).v_thresh(-0.050).v_reset(-0.065)
        .tau_m(0.010).tau_ref(0.002).c_m(250e-12).i_bg(0.0);
    auto* lif = neuron::lif_create(1, opts);

    lif->I_syn.fill_(500e-12);

    nrn::State state;
    int spike_count = 0;
    double dt = 0.0001;

    for (int step = 0; step < 1000; ++step) {
        neuron::lif_forward(lif, state, step * dt, dt);
        if (state_get(state, "spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        lif->I_syn.fill_(500e-12);
    }

    EXPECT_GT(spike_count, 5);
    EXPECT_LT(spike_count, 200);

    neuron::lif_destroy(lif);
}

TEST(LIFDynamics, SubthresholdCurrentNoSpikes) {
    // V_inf = -0.065 + 100e-12 * 0.01 / 250e-12 = -0.065 + 0.004 = -0.061 V
    // Below threshold (-0.050) — should not spike.
    auto opts = neuron::LIFOptions()
        .v_rest(-0.065).v_thresh(-0.050).v_reset(-0.065)
        .tau_m(0.010).c_m(250e-12).i_bg(0.0);
    auto* lif = neuron::lif_create(1, opts);

    lif->I_syn.fill_(100e-12);

    nrn::State state;
    int spike_count = 0;
    double dt = 0.0001;

    for (int step = 0; step < 1000; ++step) {
        neuron::lif_forward(lif, state, step * dt, dt);
        if (state_get(state, "spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        lif->I_syn.fill_(100e-12);
    }

    EXPECT_EQ(spike_count, 0);

    neuron::lif_destroy(lif);
}

TEST(LIFDynamics, VoltageRelaxesToRest) {
    auto* lif = neuron::lif_create(1);
    lif->v.fill_(-0.055);  // Above rest, below threshold

    nrn::State state;
    double dt = 0.0001;

    for (int step = 0; step < 2000; ++step) {
        neuron::lif_forward(lif, state, step * dt, dt);
    }

    float v_final = state_get(state, "v")[0].item<float>();
    EXPECT_NEAR(v_final, -0.065, 1e-5);

    neuron::lif_destroy(lif);
}

TEST(LIFDynamics, RefractoryPeriodBlocks) {
    auto opts = neuron::LIFOptions()
        .v_rest(-0.065).v_thresh(-0.050).v_reset(-0.065)
        .tau_m(0.010).tau_ref(0.005).c_m(250e-12);
    auto* lif = neuron::lif_create(1, opts);

    nrn::State state;
    double dt = 0.0001;

    // Force a spike by setting voltage above threshold.
    lif->v.fill_(-0.049);
    neuron::lif_forward(lif, state, 0.0, dt);
    EXPECT_GT(state_get(state, "spike")[0].item<float>(), 0.5f);

    // With strong current, should not spike for ~50 steps (5 ms / 0.1 ms).
    int steps_until_next_spike = -1;
    for (int step = 1; step < 200; ++step) {
        lif->I_syn.fill_(500e-12);
        neuron::lif_forward(lif, state, step * dt, dt);
        if (state_get(state, "spike")[0].item<float>() > 0.5f) {
            steps_until_next_spike = step;
            break;
        }
    }

    EXPECT_GT(steps_until_next_spike, 45);

    neuron::lif_destroy(lif);
}

TEST(LIFDynamics, MultipleNeuronsIndependent) {
    auto* lif = neuron::lif_create(100);

    // Inject current only into first 50 neurons.
    auto I = torch::zeros({100});
    I.slice(0, 0, 50).fill_(500e-12);

    nrn::State state;
    double dt = 0.0001;
    int spikes_first_half = 0;
    int spikes_second_half = 0;

    for (int step = 0; step < 500; ++step) {
        lif->I_syn.copy_(I);
        neuron::lif_forward(lif, state, step * dt, dt);
        auto s = state_get(state, "spike");
        spikes_first_half += s.slice(0, 0, 50).sum().item<int>();
        spikes_second_half += s.slice(0, 50, 100).sum().item<int>();
    }

    EXPECT_GT(spikes_first_half, 0);
    EXPECT_EQ(spikes_second_half, 0);

    neuron::lif_destroy(lif);
}

// ===========================================================================
// AdEx Dynamics
// ===========================================================================

TEST(AdExDynamics, ConstantCurrentProducesSpikes) {
    auto* adex = neuron::adex_create(1);
    adex->I_syn.fill_(1e-9);  // 1 nA — suprathreshold

    nrn::State state;
    int spike_count = 0;
    double dt = 0.0001;

    for (int step = 0; step < 2000; ++step) {
        neuron::adex_forward(adex, state, step * dt, dt);
        if (state_get(state, "spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        adex->I_syn.fill_(1e-9);
    }

    EXPECT_GT(spike_count, 1);

    neuron::adex_destroy(adex);
}

TEST(AdExDynamics, AdaptationGrows) {
    auto* adex = neuron::adex_create(1);

    nrn::State state;
    double dt = 0.0001;

    // Run with constant current for 200 ms.
    for (int step = 0; step < 2000; ++step) {
        adex->I_syn.fill_(1e-9);
        neuron::adex_forward(adex, state, step * dt, dt);
    }

    auto w_final = state_get(state, "w")[0].item<float>();
    EXPECT_GT(w_final, 0.0f);

    neuron::adex_destroy(adex);
}

// ===========================================================================
// Izhikevich Dynamics
// ===========================================================================

TEST(IzhikevichDynamics, RegularSpikingWithInput) {
    auto* izh = neuron::izh_create(1);
    izh->I_syn.fill_(10.0);  // Dimensionless, moderate input

    nrn::State state;
    int spike_count = 0;
    double dt = 0.001;  // 1 ms

    for (int step = 0; step < 1000; ++step) {
        neuron::izh_forward(izh, state, step * dt, dt);
        if (state_get(state, "spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        izh->I_syn.fill_(10.0);
    }

    EXPECT_GT(spike_count, 5);

    neuron::izh_destroy(izh);
}

TEST(IzhikevichDynamics, NoInputNoSpikes) {
    auto* izh = neuron::izh_create(1);

    nrn::State state;
    double dt = 0.001;

    for (int step = 0; step < 100; ++step) {
        neuron::izh_forward(izh, state, step * dt, dt);
        EXPECT_LT(state_get(state, "spike")[0].item<float>(), 0.5f);
    }

    neuron::izh_destroy(izh);
}

// ===========================================================================
// Connection::deliver()
// ===========================================================================

TEST(ConnectionDeliver, ProducesNonzeroISyn) {
    auto* lif = neuron::lif_create(32);
    auto module = neuron::lif_as_module(lif);
    auto* pop_raw = population_create("neurons", module, 32, torch::kCPU);
    auto pop = std::shared_ptr<Population>(pop_raw, population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.5));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_module = synapse::static_synapse_as_module(syn);

    auto conn = connect(pop, pop, &gen, syn_module,
                        ConnectOptions().block_size(16));

    // Initialize state.
    nrn::State state;
    neuron::lif_forward(lif, state, 0.0, 0.0);
    pop->state = state;

    // Create spikes: neuron 0 fires.
    auto spikes = torch::zeros({32});
    spikes[0] = 1.0;

    connection_deliver(conn.get(), spikes, 0.0, 0.0001);

    auto I_syn_total = state_get(pop->state, "I_syn").abs().sum().item<float>();
    EXPECT_GT(I_syn_total, 0.0f);

    random_topology_destroy(topo);
    synapse::static_synapse_destroy(syn);
    neuron::lif_destroy(lif);
}

// ===========================================================================
// Simulation::step()
// ===========================================================================

TEST(SimulationStep, RunsWithoutCrashing) {
    auto* lif = neuron::lif_create(64);
    auto module = neuron::lif_as_module(lif);
    auto* pop_raw = population_create("exc", module, 64, torch::kCPU);
    auto exc = std::shared_ptr<Population>(pop_raw, population_destroy);

    auto* region = region_create("test");
    region_add_population(region, exc);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.0001).dt_slow(0.01).dt_structural(1.0)
        .duration(0.01).device(torch::kCPU);

    auto* sim = sim_create(region, sim_opts);
    sim_run_steps(sim, 10);
    SUCCEED();

    sim_destroy(sim);
    region_destroy(region);
    neuron::lif_destroy(lif);
}

TEST(SimulationStep, RunsWithConnections) {
    auto* lif = neuron::lif_create(32, neuron::LIFOptions().i_bg(500e-12));
    auto module = neuron::lif_as_module(lif);
    auto* pop_raw = population_create("neurons", module, 32, torch::kCPU);
    auto pop = std::shared_ptr<Population>(pop_raw, population_destroy);

    auto* topo = random_topology_create(RandomTopologyOptions().probability(0.1));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_module = synapse::static_synapse_as_module(syn);

    auto conn = connect(pop, pop, &gen, syn_module,
                        ConnectOptions().block_size(16));

    auto* region = region_create("test");
    region_add_population(region, pop);
    region_add_connection(region, conn);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.0001).dt_slow(0.01).dt_structural(1.0)
        .device(torch::kCPU);

    auto* sim = sim_create(region, sim_opts);
    sim_record(sim, pop, {"v", "spike"});
    sim_run_steps(sim, 500);  // 50 ms — enough for LIF to reach threshold from rest

    auto spikes = sim_get_spikes(sim, pop);
    // With i_bg = 500 pA, neurons should spike within ~16 ms.
    EXPECT_GT(spikes.size(0), 0);

    sim_destroy(sim);
    region_destroy(region);
    random_topology_destroy(topo);
    synapse::static_synapse_destroy(syn);
    neuron::lif_destroy(lif);
}

TEST(SimulationStep, NoNaNOrInf) {
    auto* lif = neuron::lif_create(32, neuron::LIFOptions().i_bg(500e-12));
    auto module = neuron::lif_as_module(lif);
    auto* pop_raw = population_create("neurons", module, 32, torch::kCPU);
    auto pop = std::shared_ptr<Population>(pop_raw, population_destroy);

    auto* region = region_create("test");
    region_add_population(region, pop);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.0001).dt_slow(0.01).dt_structural(1.0)
        .device(torch::kCPU);

    auto* sim = sim_create(region, sim_opts);
    sim_run_steps(sim, 100);

    EXPECT_FALSE(torch::any(torch::isnan(state_get(pop->state, "v"))).item<bool>());
    EXPECT_FALSE(torch::any(torch::isinf(state_get(pop->state, "v"))).item<bool>());

    sim_destroy(sim);
    region_destroy(region);
    neuron::lif_destroy(lif);
}
