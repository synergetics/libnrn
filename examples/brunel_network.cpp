/// Brunel Balanced E-I Network Example
///
/// A balanced excitatory-inhibitory (E-I) spiking network based on Brunel
/// (2000).  Uses LIF neurons with proper E/I weight balance (g=5)
/// and background current to approximate external Poisson input.
///
/// Usage: brunel_network [cpu|cuda] [neuron_count]
///   device:       cpu or cuda (default: cpu)
///   neuron_count: total neurons, split 4:1 E:I (default: 5000)

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

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

int main(int argc, char* argv[]) {
    // -----------------------------------------------------------------------
    // Parse CLI args
    // -----------------------------------------------------------------------
    torch::Device device = torch::kCPU;
    int64_t N_total = 5000;

    if (argc >= 2) {
        std::string dev_str = argv[1];
        if (dev_str == "cuda" || dev_str == "CUDA" || dev_str == "gpu") {
            if (torch::cuda::is_available()) {
                device = torch::kCUDA;
            } else {
                std::cerr << "CUDA requested but not available. Falling back to CPU.\n";
            }
        }
    }
    if (argc >= 3) {
        N_total = std::atol(argv[2]);
        if (N_total < 5) N_total = 5;
    }

    int64_t N_exc = (N_total * 4) / 5;
    int64_t N_inh = N_total - N_exc;

    std::cout << "Using device: " << device << "\n";
    std::cout << "Neurons: " << N_total << " (" << N_exc << " E + " << N_inh << " I)\n";

    // -----------------------------------------------------------------------
    // Brunel (2000) parameters
    // -----------------------------------------------------------------------
    const double  conn_prob = 0.1;
    const double J_E = 500.0e-12;
    const double g   = 5.0;
    const double J_I = -g * J_E;
    const double i_bg_exc = 380.0e-12;
    const double i_bg_inh = 380.0e-12;
    const double sim_duration = 0.2;
    const double dt = 0.1e-3;

    // -----------------------------------------------------------------------
    // 1. Create neuron populations
    // -----------------------------------------------------------------------
    auto exc_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(20.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(i_bg_exc);

    auto inh_opts = neuron::LIFOptions()
        .v_rest(-65.0_mV).v_thresh(-50.0_mV).v_reset(-65.0_mV)
        .tau_m(10.0_ms).tau_ref(2.0_ms).c_m(250.0_pF).i_bg(i_bg_inh);

    auto* exc_lif = neuron::lif_create(N_exc, exc_opts);
    auto* inh_lif = neuron::lif_create(N_inh, inh_opts);

    auto exc = std::shared_ptr<Population>(
        population_create("excitatory", neuron::lif_as_module(exc_lif), N_exc, device),
        population_destroy);
    auto inh = std::shared_ptr<Population>(
        population_create("inhibitory", neuron::lif_as_module(inh_lif), N_inh, device),
        population_destroy);

    // -----------------------------------------------------------------------
    // 2. Create connectivity
    // -----------------------------------------------------------------------
    auto* topo = random_topology_create(RandomTopologyOptions()
        .probability(conn_prob).allow_autapses(false));
    auto gen = random_topology_as_generator(topo);

    auto* syn = synapse::static_synapse_create(1);
    auto syn_mod = synapse::static_synapse_as_module(syn);

    auto co_ee = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_E);
    auto co_ei = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_E);
    auto co_ie = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_I);
    auto co_ii = ConnectOptions().block_size(256).representation(BlockDense)
        .default_delay(0.001).weight_scale(J_I);

    auto ee_conn = connect(exc, exc, &gen, syn_mod, co_ee);
    auto ei_conn = connect(exc, inh, &gen, syn_mod, co_ei);
    auto ie_conn = connect(inh, exc, &gen, syn_mod, co_ie);
    auto ii_conn = connect(inh, inh, &gen, syn_mod, co_ii);

    std::cout << "Created 4 connection pathways (E->E, E->I, I->E, I->I)\n";

    // -----------------------------------------------------------------------
    // 3. Attach STDP to E->E
    // -----------------------------------------------------------------------
    auto* stdp = stdp_create(STDPOptions()
        .tau_plus(20.0_ms).tau_minus(20.0_ms)
        .a_plus(0.01).a_minus(-0.012)
        .learning_rate(0.001).w_min(0.0).w_max(J_E * 5.0));
    connection_attach(ee_conn.get(), stdp_as_rule(stdp));

    // -----------------------------------------------------------------------
    // 4. Compose region
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
    // 5. Run simulation
    // -----------------------------------------------------------------------
    auto sim_opts = SimulationOptions()
        .dt_fast(dt).dt_slow(10.0_ms).dt_structural(1.0_s)
        .duration(sim_duration).device(device);

    auto* sim = sim_create(region, sim_opts);

    sim_record(sim, exc, {"v", "spike"}, /*subsample=*/10);
    sim_record(sim, inh, {"v", "spike"}, /*subsample=*/10);

    int64_t n_steps = static_cast<int64_t>(sim_duration / dt);

    std::cout << "Running simulation for " << sim_duration
              << " s (" << n_steps << " steps) at dt = " << dt * 1e3 << " ms...\n";

    // Synchronize before timing on GPU.
    if (device == torch::kCUDA) {
        torch::cuda::synchronize();
    }
    auto t_start = std::chrono::steady_clock::now();

    sim_run(sim);

    if (device == torch::kCUDA) {
        torch::cuda::synchronize();
    }
    auto t_end = std::chrono::steady_clock::now();

    double wall_time = std::chrono::duration<double>(t_end - t_start).count();

    // -----------------------------------------------------------------------
    // 6. Results
    // -----------------------------------------------------------------------
    auto exc_spikes = sim_get_spikes(sim, exc);
    auto inh_spikes = sim_get_spikes(sim, inh);

    int64_t n_exc_spikes = exc_spikes.size(0);
    int64_t n_inh_spikes = inh_spikes.size(0);

    double exc_rate = static_cast<double>(n_exc_spikes) / (N_exc * sim_duration);
    double inh_rate = static_cast<double>(n_inh_spikes) / (N_inh * sim_duration);
    double throughput = static_cast<double>(N_total) * n_steps / wall_time;

    std::cout << "\n--- Results ---\n";
    std::cout << "Excitatory spikes: " << n_exc_spikes
              << " (mean rate: " << exc_rate << " Hz)\n";
    std::cout << "Inhibitory spikes: " << n_inh_spikes
              << " (mean rate: " << inh_rate << " Hz)\n";
    std::cout << "Wall time: " << wall_time << " s\n";
    std::cout << "Throughput: " << throughput / 1e6 << " M neuron-timesteps/sec\n";

    if (exc_rate >= 1.0 && exc_rate <= 50.0 &&
        inh_rate >= 1.0 && inh_rate <= 100.0) {
        std::cout << "Firing rates are in plausible biological range.\n";
    } else {
        std::cout << "WARNING: Firing rates outside expected range.\n";
    }

    // E->E weight stats after STDP.
    auto ee_w = ee_conn->connectivity.weights.to(torch::kCPU);
    std::cout << "E->E weight mean: " << ee_w.mean().item<float>()
              << ", std: " << ee_w.std().item<float>() << "\n";

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
