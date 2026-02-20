/// Brunel Balanced E-I Network Example
///
/// This is an aspirational example demonstrating the target libnrn API for
/// a balanced excitatory-inhibitory (E-I) spiking network based on Brunel
/// (2000).  The network consists of:
///
///   - 10,000 excitatory LIF neurons
///   -  2,500 inhibitory LIF neurons
///
/// connected with random (Erdos-Renyi) topology and pair-based STDP on
/// excitatory-to-excitatory connections.
///
/// NOTE: This code is designed to compile against the skeleton headers and
/// illustrate the API.  Full simulation requires completed implementations
/// of all referenced modules.

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
    // Select device: prefer CUDA if available.
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA, 0)
        : torch::kCPU;
    std::cout << "Using device: " << device << "\n";

    // -----------------------------------------------------------------------
    // 1. Create neuron populations
    // -----------------------------------------------------------------------

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

    auto exc_module = neuron::LIF(10000, exc_opts);
    auto inh_module = neuron::LIF(2500, inh_opts);

    auto exc = std::make_shared<Population>(
        "excitatory",
        exc_module.ptr(),
        10000,
        device);

    auto inh = std::make_shared<Population>(
        "inhibitory",
        inh_module.ptr(),
        2500,
        device);

    std::cout << "Created populations: "
              << exc->name() << " (" << exc->size() << "), "
              << inh->name() << " (" << inh->size() << ")\n";

    // -----------------------------------------------------------------------
    // 2. Create connectivity
    // -----------------------------------------------------------------------

    auto random_topo = Random(RandomTopologyOptions()
        .probability(0.1)
        .allow_autapses(false));

    auto conn_opts = ConnectOptions()
        .block_size(256)
        .representation(BlockDense)
        .default_delay(0.001);

    auto syn_static = StaticSynapse(1);  // Placeholder synapse module.

    // E -> E
    auto ee_conn = connect(exc, exc, random_topo,
                           syn_static.ptr(), conn_opts);

    // E -> I
    auto ei_conn = connect(exc, inh, random_topo,
                           syn_static.ptr(), conn_opts);

    // I -> E
    auto ie_conn = connect(inh, exc, random_topo,
                           syn_static.ptr(), conn_opts);

    // I -> I
    auto ii_conn = connect(inh, inh, random_topo,
                           syn_static.ptr(), conn_opts);

    std::cout << "Created 4 connection pathways (E->E, E->I, I->E, I->I)\n";

    // -----------------------------------------------------------------------
    // 3. Attach STDP to excitatory connections
    // -----------------------------------------------------------------------

    auto stdp_rule = std::make_shared<STDP>(STDPOptions()
        .tau_plus(20.0_ms)
        .tau_minus(20.0_ms)
        .a_plus(0.01)
        .a_minus(-0.012)
        .learning_rate(1.0));

    ee_conn->attach(stdp_rule);

    std::cout << "Attached STDP to E->E connections\n";

    // -----------------------------------------------------------------------
    // 4. Compose into a Region
    // -----------------------------------------------------------------------

    Region region("brunel_network");
    region->add(exc, inh);
    region->add(ee_conn, ei_conn, ie_conn, ii_conn);

    std::cout << "Region total neurons: " << region->total_size() << "\n";

    // -----------------------------------------------------------------------
    // 5. Configure and run simulation
    // -----------------------------------------------------------------------

    auto sim_opts = SimulationOptions()
        .dt_fast(0.1_ms)
        .dt_slow(10.0_ms)
        .dt_structural(1.0_s)
        .duration(1.0_s)
        .device(device);

    Simulation sim(region, sim_opts);

    // Record spikes from both populations.
    sim.record(exc, {"v", "spike"}, /*subsample=*/10);
    sim.record(inh, {"v", "spike"}, /*subsample=*/10);

    std::cout << "Running simulation for 1.0 s at dt = 0.1 ms...\n";
    sim.run();
    std::cout << "Simulation complete.\n";

    // -----------------------------------------------------------------------
    // 6. Retrieve and display results
    // -----------------------------------------------------------------------

    auto exc_spikes = sim.get_spikes(exc);
    auto inh_spikes = sim.get_spikes(inh);

    int64_t n_exc_spikes = exc_spikes.size(0);
    int64_t n_inh_spikes = inh_spikes.size(0);

    double sim_duration = 1.0;  // seconds
    double exc_rate = static_cast<double>(n_exc_spikes) / (10000.0 * sim_duration);
    double inh_rate = static_cast<double>(n_inh_spikes) / (2500.0 * sim_duration);

    std::cout << "\n--- Results ---\n";
    std::cout << "Excitatory spikes: " << n_exc_spikes
              << " (mean rate: " << exc_rate << " Hz)\n";
    std::cout << "Inhibitory spikes: " << n_inh_spikes
              << " (mean rate: " << inh_rate << " Hz)\n";

    // Save to file.
    sim.save("brunel_output.h5");
    std::cout << "Saved results to brunel_output.h5\n";

    return 0;
}
