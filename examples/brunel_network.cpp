/// Brunel Balanced E-I Network Example
///
/// A balanced excitatory-inhibitory (E-I) spiking network based on Brunel
/// (2000).  Uses 4000 excitatory + 1000 inhibitory LIF neurons with proper
/// E/I weight balance (g=5) and background current to approximate external
/// Poisson input.

#include <chrono>
#include <iostream>
#include <memory>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/neuron/lif.h>
#include <nrn/synapse/static.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/plasticity/stdp.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>
#include <nrn/simulation/options.h>

using namespace nrn;
using namespace nrn::literals;

int main() {
    torch::Device device = torch::kCPU;
    std::cout << "Using device: " << device << "\n";

    // -----------------------------------------------------------------------
    // Brunel (2000) parameters
    // -----------------------------------------------------------------------

    const int64_t N_exc = 4000;
    const int64_t N_inh = 1000;
    const double  conn_prob = 0.1;

    // Weight parameters
    // PSP amplitude ~ 0.1 mV  →  weight = PSP * C_m / dt = 0.1mV * 250pF / 0.1ms = 250 pA
    // Topology generates U(0,1) weights; weight_scale sets the upper bound.
    // Mean weight = weight_scale / 2 ≈ 250 pA → weight_scale = 500 pA.
    const double J_E = 500.0e-12;   // excitatory weight_scale (500 pA → mean 250 pA)
    const double g   = 5.0;         // inhibition dominance factor
    const double J_I = -g * J_E;    // inhibitory weight_scale (negative, 5x stronger)

    // Background current to approximate external Poisson input.
    // Threshold current: i_thr = (v_thresh - v_rest) * c_m / tau_m = 15mV * 250pF / 20ms = 187.5 pA
    // Use η ≈ 2 (suprathreshold) for asynchronous irregular regime.
    const double i_bg_exc = 380.0e-12;  // ~2× threshold
    const double i_bg_inh = 380.0e-12;

    const double sim_duration = 0.2;  // 200 ms (fast for testing)
    const double dt = 0.1e-3;         // 0.1 ms

    // -----------------------------------------------------------------------
    // 1. Create neuron populations
    // -----------------------------------------------------------------------

    auto exc_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV)
        .v_thresh(-50.0_mV)
        .v_reset(-65.0_mV)
        .tau_m(20.0_ms)
        .tau_ref(2.0_ms)
        .c_m(250.0_pF)
        .i_bg(i_bg_exc);

    auto inh_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV)
        .v_thresh(-50.0_mV)
        .v_reset(-65.0_mV)
        .tau_m(10.0_ms)
        .tau_ref(2.0_ms)
        .c_m(250.0_pF)
        .i_bg(i_bg_inh);

    auto* exc_lif = neuron::lif_create(N_exc, exc_opts);
    auto* inh_lif = neuron::lif_create(N_inh, inh_opts);

    auto exc = std::shared_ptr<Population>(
        population_create("excitatory", neuron::lif_as_module(exc_lif), N_exc, device),
        population_destroy);

    auto inh = std::shared_ptr<Population>(
        population_create("inhibitory", neuron::lif_as_module(inh_lif), N_inh, device),
        population_destroy);

    std::cout << "Created populations: "
              << exc->name << " (" << exc->n << "), "
              << inh->name << " (" << inh->n << ")\n";

    // -----------------------------------------------------------------------
    // 2. Create connectivity with E/I weight balance
    // -----------------------------------------------------------------------

    auto* topo = random_topology_create(RandomTopologyOptions()
        .probability(conn_prob)
        .allow_autapses(false));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto conn_opts_ee = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_E);
    auto conn_opts_ei = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_E);
    auto conn_opts_ie = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_I);
    auto conn_opts_ii = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_I);

    auto ee_conn = connect(exc, exc, &gen, syn_mod, conn_opts_ee);
    auto ei_conn = connect(exc, inh, &gen, syn_mod, conn_opts_ei);
    auto ie_conn = connect(inh, exc, &gen, syn_mod, conn_opts_ie);
    auto ii_conn = connect(inh, inh, &gen, syn_mod, conn_opts_ii);

    std::cout << "Created 4 connection pathways (E->E, E->I, I->E, I->I)\n";
    std::cout << "  J_E = " << J_E << ", J_I = " << J_I << " (g=" << g << ")\n";

    // -----------------------------------------------------------------------
    // 3. Attach STDP to E->E connections
    // -----------------------------------------------------------------------

    auto* stdp = stdp_create(STDPOptions()
        .tau_plus(20.0_ms)
        .tau_minus(20.0_ms)
        .a_plus(0.01)
        .a_minus(-0.012)
        .learning_rate(0.001)
        .w_min(0.0)
        .w_max(J_E * 5.0));

    connection_attach(ee_conn.get(), stdp_as_rule(stdp));

    std::cout << "Attached STDP to E->E connections\n";

    // -----------------------------------------------------------------------
    // 4. Compose into a Region
    // -----------------------------------------------------------------------

    auto* region = region_create("brunel_network");
    region_add_population(region, exc);
    region_add_population(region, inh);
    region_add_connection(region, ee_conn);
    region_add_connection(region, ei_conn);
    region_add_connection(region, ie_conn);
    region_add_connection(region, ii_conn);

    std::cout << "Region total neurons: " << region_total_size(region) << "\n";

    // -----------------------------------------------------------------------
    // 5. Configure and run simulation
    // -----------------------------------------------------------------------

    auto sim_opts = SimulationOptions()
        .dt_fast(dt)
        .dt_slow(10.0_ms)
        .dt_structural(1.0_s)
        .duration(sim_duration)
        .device(device);

    auto* sim = sim_create(region, sim_opts);

    sim_record(sim, exc, {"v", "spike"}, /*subsample=*/10);
    sim_record(sim, inh, {"v", "spike"}, /*subsample=*/10);

    std::cout << "Running simulation for " << sim_duration
              << " s at dt = " << dt * 1e3 << " ms...\n";

    auto t_start = std::chrono::steady_clock::now();
    sim_run(sim);
    auto t_end = std::chrono::steady_clock::now();

    double wall_time = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Simulation complete in " << wall_time << " s wall time.\n";

    // -----------------------------------------------------------------------
    // 6. Retrieve and display results
    // -----------------------------------------------------------------------

    auto exc_spikes = sim_get_spikes(sim, exc);
    auto inh_spikes = sim_get_spikes(sim, inh);

    int64_t n_exc_spikes = exc_spikes.size(0);
    int64_t n_inh_spikes = inh_spikes.size(0);

    double exc_rate = static_cast<double>(n_exc_spikes) / (N_exc * sim_duration);
    double inh_rate = static_cast<double>(n_inh_spikes) / (N_inh * sim_duration);

    std::cout << "\n--- Results ---\n";
    std::cout << "Excitatory spikes: " << n_exc_spikes
              << " (mean rate: " << exc_rate << " Hz)\n";
    std::cout << "Inhibitory spikes: " << n_inh_spikes
              << " (mean rate: " << inh_rate << " Hz)\n";

    if (exc_rate >= 1.0 && exc_rate <= 50.0 &&
        inh_rate >= 1.0 && inh_rate <= 100.0) {
        std::cout << "Firing rates are in plausible biological range.\n";
    } else {
        std::cout << "WARNING: Firing rates outside expected range (1-50 Hz exc, 1-100 Hz inh).\n";
    }

    // Check E->E weight distribution after STDP
    auto ee_weights = ee_conn->connectivity.weights;
    std::cout << "E->E weight mean: " << ee_weights.mean().item<float>()
              << ", std: " << ee_weights.std().item<float>() << "\n";

    // -----------------------------------------------------------------------
    // 7. Cleanup
    // -----------------------------------------------------------------------

    sim_destroy(sim);
    region_destroy(region);
    ee_conn.reset(); ei_conn.reset(); ie_conn.reset(); ii_conn.reset();
    exc.reset(); inh.reset();
    stdp_destroy(stdp);
    synapse::static_synapse_destroy(syn);
    random_topology_destroy(topo);
    neuron::lif_destroy(exc_lif);
    neuron::lif_destroy(inh_lif);

    return 0;
}
