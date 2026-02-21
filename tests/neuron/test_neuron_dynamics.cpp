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
    neuron::LIF lif(1, opts);

    lif->I_syn.fill_(500e-12);

    nrn::State state;
    int spike_count = 0;
    double dt = 0.0001;

    for (int step = 0; step < 1000; ++step) {
        lif->forward(state, step * dt, dt);
        if (state.get("spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        lif->I_syn.fill_(500e-12);
    }

    EXPECT_GT(spike_count, 5);
    EXPECT_LT(spike_count, 200);
}

TEST(LIFDynamics, SubthresholdCurrentNoSpikes) {
    // V_inf = -0.065 + 100e-12 * 0.01 / 250e-12 = -0.065 + 0.004 = -0.061 V
    // Below threshold (-0.050) — should not spike.
    auto opts = neuron::LIFOptions()
        .v_rest(-0.065).v_thresh(-0.050).v_reset(-0.065)
        .tau_m(0.010).c_m(250e-12).i_bg(0.0);
    neuron::LIF lif(1, opts);

    lif->I_syn.fill_(100e-12);

    nrn::State state;
    int spike_count = 0;
    double dt = 0.0001;

    for (int step = 0; step < 1000; ++step) {
        lif->forward(state, step * dt, dt);
        if (state.get("spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        lif->I_syn.fill_(100e-12);
    }

    EXPECT_EQ(spike_count, 0);
}

TEST(LIFDynamics, VoltageRelaxesToRest) {
    neuron::LIF lif(1);
    lif->v.fill_(-0.055);  // Above rest, below threshold

    nrn::State state;
    double dt = 0.0001;

    for (int step = 0; step < 2000; ++step) {
        lif->forward(state, step * dt, dt);
    }

    float v_final = state.get("v")[0].item<float>();
    EXPECT_NEAR(v_final, -0.065, 1e-5);
}

TEST(LIFDynamics, RefractoryPeriodBlocks) {
    auto opts = neuron::LIFOptions()
        .v_rest(-0.065).v_thresh(-0.050).v_reset(-0.065)
        .tau_m(0.010).tau_ref(0.005).c_m(250e-12);
    neuron::LIF lif(1, opts);

    nrn::State state;
    double dt = 0.0001;

    // Force a spike by setting voltage above threshold.
    lif->v.fill_(-0.049);
    lif->forward(state, 0.0, dt);
    EXPECT_GT(state.get("spike")[0].item<float>(), 0.5f);

    // With strong current, should not spike for ~50 steps (5 ms / 0.1 ms).
    int steps_until_next_spike = -1;
    for (int step = 1; step < 200; ++step) {
        lif->I_syn.fill_(500e-12);
        lif->forward(state, step * dt, dt);
        if (state.get("spike")[0].item<float>() > 0.5f) {
            steps_until_next_spike = step;
            break;
        }
    }

    EXPECT_GT(steps_until_next_spike, 45);
}

TEST(LIFDynamics, MultipleNeuronsIndependent) {
    neuron::LIF lif(100);

    // Inject current only into first 50 neurons.
    auto I = torch::zeros({100});
    I.slice(0, 0, 50).fill_(500e-12);

    nrn::State state;
    double dt = 0.0001;
    int spikes_first_half = 0;
    int spikes_second_half = 0;

    for (int step = 0; step < 500; ++step) {
        lif->I_syn.copy_(I);
        lif->forward(state, step * dt, dt);
        auto s = state.get("spike");
        spikes_first_half += s.slice(0, 0, 50).sum().item<int>();
        spikes_second_half += s.slice(0, 50, 100).sum().item<int>();
    }

    EXPECT_GT(spikes_first_half, 0);
    EXPECT_EQ(spikes_second_half, 0);
}

// ===========================================================================
// AdEx Dynamics
// ===========================================================================

TEST(AdExDynamics, ConstantCurrentProducesSpikes) {
    neuron::AdEx adex(1);
    adex->I_syn.fill_(1e-9);  // 1 nA — suprathreshold

    nrn::State state;
    int spike_count = 0;
    double dt = 0.0001;

    for (int step = 0; step < 2000; ++step) {
        adex->forward(state, step * dt, dt);
        if (state.get("spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        adex->I_syn.fill_(1e-9);
    }

    EXPECT_GT(spike_count, 1);
}

TEST(AdExDynamics, AdaptationGrows) {
    neuron::AdEx adex(1);

    nrn::State state;
    double dt = 0.0001;

    // Run with constant current for 200 ms.
    for (int step = 0; step < 2000; ++step) {
        adex->I_syn.fill_(1e-9);
        adex->forward(state, step * dt, dt);
    }

    auto w_final = state.get("w")[0].item<float>();
    EXPECT_GT(w_final, 0.0f);
}

// ===========================================================================
// Izhikevich Dynamics
// ===========================================================================

TEST(IzhikevichDynamics, RegularSpikingWithInput) {
    neuron::Izhikevich izh(1);
    izh->I_syn.fill_(10.0);  // Dimensionless, moderate input

    nrn::State state;
    int spike_count = 0;
    double dt = 0.001;  // 1 ms

    for (int step = 0; step < 1000; ++step) {
        izh->forward(state, step * dt, dt);
        if (state.get("spike")[0].item<float>() > 0.5f) {
            spike_count++;
        }
        izh->I_syn.fill_(10.0);
    }

    EXPECT_GT(spike_count, 5);
}

TEST(IzhikevichDynamics, NoInputNoSpikes) {
    neuron::Izhikevich izh(1);

    nrn::State state;
    double dt = 0.001;

    for (int step = 0; step < 100; ++step) {
        izh->forward(state, step * dt, dt);
        EXPECT_LT(state.get("spike")[0].item<float>(), 0.5f);
    }
}

// ===========================================================================
// Connection::deliver()
// ===========================================================================

TEST(ConnectionDeliver, ProducesNonzeroISyn) {
    auto lif = neuron::LIF(32);
    auto pop = std::make_shared<Population>(
        "neurons", lif.ptr(), 32, torch::kCPU);

    auto topo = Random(RandomTopologyOptions().probability(0.5));
    auto syn = neuron::LIF(1);  // Unused but needed for connect()
    auto conn = connect(pop, pop, topo, syn.ptr(),
                        ConnectOptions().block_size(16));

    // Initialize state.
    nrn::State state;
    lif->forward(state, 0.0, 0.0);
    pop->state() = state;

    // Create spikes: neuron 0 fires.
    auto spikes = torch::zeros({32});
    spikes[0] = 1.0;

    conn->deliver(spikes, 0.0, 0.0001);

    auto I_syn_total = pop->state().get("I_syn").abs().sum().item<float>();
    EXPECT_GT(I_syn_total, 0.0f);
}

// ===========================================================================
// Simulation::step()
// ===========================================================================

TEST(SimulationStep, RunsWithoutCrashing) {
    auto lif_mod = neuron::LIF(64);
    auto exc = std::make_shared<Population>(
        "exc", lif_mod.ptr(), 64, torch::kCPU);

    Region region("test");
    region->add(exc);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.0001).dt_slow(0.01).dt_structural(1.0)
        .duration(0.01).device(torch::kCPU);

    Simulation sim(region, sim_opts);
    sim.run_steps(10);
    SUCCEED();
}

TEST(SimulationStep, RunsWithConnections) {
    auto lif_mod = neuron::LIF(32, neuron::LIFOptions().i_bg(500e-12));
    auto pop = std::make_shared<Population>(
        "neurons", lif_mod.ptr(), 32, torch::kCPU);

    auto topo = Random(RandomTopologyOptions().probability(0.1));
    auto syn = neuron::LIF(1);  // Placeholder module for connect()
    auto conn = connect(pop, pop, topo, syn.ptr(),
                        ConnectOptions().block_size(16));

    Region region("test");
    region->add(pop);
    region->add(conn);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.0001).dt_slow(0.01).dt_structural(1.0)
        .device(torch::kCPU);

    Simulation sim(region, sim_opts);
    sim.record(pop, {"v", "spike"});
    sim.run_steps(500);  // 50 ms — enough for LIF to reach threshold from rest

    auto spikes = sim.get_spikes(pop);
    // With i_bg = 500 pA, neurons should spike within ~16 ms.
    EXPECT_GT(spikes.size(0), 0);
}

TEST(SimulationStep, NoNaNOrInf) {
    auto lif_mod = neuron::LIF(32, neuron::LIFOptions().i_bg(500e-12));
    auto pop = std::make_shared<Population>(
        "neurons", lif_mod.ptr(), 32, torch::kCPU);

    Region region("test");
    region->add(pop);

    auto sim_opts = SimulationOptions()
        .dt_fast(0.0001).dt_slow(0.01).dt_structural(1.0)
        .device(torch::kCPU);

    Simulation sim(region, sim_opts);
    sim.run_steps(100);

    auto& s = pop->state();
    EXPECT_FALSE(torch::any(torch::isnan(s.get("v"))).item<bool>());
    EXPECT_FALSE(torch::any(torch::isinf(s.get("v"))).item<bool>());
}
