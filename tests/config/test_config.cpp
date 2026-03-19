#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include <torch/torch.h>

#include <nrn/config/config.h>
#include <nrn/connectivity/connect.h>
#include <nrn/plasticity/plasticity_rule.h>
#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>

using namespace nrn;

// Helper: write a YAML string to a temp file and return its path.
static std::string write_temp_yaml(const std::string& content,
                                    const std::string& name) {
    auto path = std::string("/tmp/libnrn_test_") + name + ".yaml";
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

// ---------------------------------------------------------------------------
// Load a region from YAML.
// ---------------------------------------------------------------------------

TEST(Config, LoadRegion) {
    auto yaml = R"(
region_name: test_region
populations:
  - name: excitatory
    model: LIF
    count: 100
    params:
      v_rest: -0.065
      v_thresh: -0.050
      tau_m: 0.020
      i_bg: 380.0e-12
  - name: inhibitory
    model: LIF
    count: 25
    params:
      v_rest: -0.065
      v_thresh: -0.050
      tau_m: 0.010
      i_bg: 380.0e-12
connections:
  - source: excitatory
    target: inhibitory
    topology: random
    topology_params:
      probability: 0.1
    block_size: 32
    weight_scale: 500.0e-12
)";

    auto path = write_temp_yaml(yaml, "load_region");
    auto* region = config::load_region(path);

    ASSERT_NE(region, nullptr);
    EXPECT_EQ(region->name, "test_region");
    EXPECT_EQ(region->populations.size(), 2u);
    EXPECT_EQ(region->connections.size(), 1u);
    EXPECT_EQ(region_total_size(region), 125);

    auto exc = region_find_population(region, "excitatory");
    ASSERT_NE(exc, nullptr);
    EXPECT_EQ(exc->n, 100);

    auto inh = region_find_population(region, "inhibitory");
    ASSERT_NE(inh, nullptr);
    EXPECT_EQ(inh->n, 25);

    region_destroy(region);
}

// ---------------------------------------------------------------------------
// Load a full simulation.
// ---------------------------------------------------------------------------

TEST(Config, LoadSimulation) {
    auto yaml = R"(
simulation:
  duration: 0.01
  dt_fast: 0.0001
  device: cpu
populations:
  - name: neurons
    model: LIF
    count: 50
    params:
      v_rest: -0.065
      v_thresh: -0.050
      tau_m: 0.020
      i_bg: 400.0e-12
connections:
  - source: neurons
    target: neurons
    topology: random
    topology_params:
      probability: 0.2
    block_size: 16
    weight_scale: 500.0e-12
)";

    auto path = write_temp_yaml(yaml, "load_sim");
    auto* sim = config::load(path);

    ASSERT_NE(sim, nullptr);
    EXPECT_EQ(sim->region->populations.size(), 1u);
    EXPECT_EQ(sim->region->connections.size(), 1u);
    EXPECT_DOUBLE_EQ(sim->options.duration(), 0.01);
    EXPECT_DOUBLE_EQ(sim->options.dt_fast(), 0.0001);

    // Run it — should not crash.
    sim_run(sim);

    sim_destroy(sim);
}

// ---------------------------------------------------------------------------
// Invalid model name should throw.
// ---------------------------------------------------------------------------

TEST(Config, InvalidModel_Throws) {
    auto yaml = R"(
populations:
  - name: bad
    model: FooBarNeuron
    count: 10
)";

    auto path = write_temp_yaml(yaml, "invalid_model");
    EXPECT_THROW(config::load_region(path), c10::Error);
}

// ---------------------------------------------------------------------------
// STDP attachment via config.
// ---------------------------------------------------------------------------

TEST(Config, STDPAttachment) {
    auto yaml = R"(
populations:
  - name: exc
    model: LIF
    count: 64
    params:
      v_rest: -0.065
      v_thresh: -0.050
      tau_m: 0.020
      i_bg: 400.0e-12
connections:
  - source: exc
    target: exc
    topology: random
    topology_params:
      probability: 0.2
    block_size: 16
    weight_scale: 500.0e-12
    plasticity: stdp
    plasticity_params:
      tau_plus: 0.020
      a_plus: 0.01
      a_minus: -0.012
)";

    auto path = write_temp_yaml(yaml, "stdp_attach");
    auto* region = config::load_region(path);

    ASSERT_NE(region, nullptr);
    EXPECT_EQ(region->connections.size(), 1u);
    // Connection should have one plasticity rule attached.
    EXPECT_EQ(region->connections[0]->plasticity_rules.size(), 1u);

    region_destroy(region);
}
