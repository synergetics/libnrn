#include <nrn/config/config.h>

#include <yaml-cpp/yaml.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/neuron/lif.h>
#include <nrn/neuron/adex.h>
#include <nrn/neuron/izhikevich.h>
#include <nrn/synapse/static.h>
#include <nrn/connectivity/connect.h>
#include <nrn/connectivity/options.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/connectivity/topology/distance_dependent.h>
#include <nrn/plasticity/stdp.h>
#include <nrn/network/population.h>
#include <nrn/network/region.h>
#include <nrn/simulation/runner.h>
#include <nrn/simulation/options.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace nrn {
namespace config {

// ---------------------------------------------------------------------------
// Helpers: parse typed option structs from YAML nodes
// ---------------------------------------------------------------------------

static neuron::LIFOptions parse_lif_options(const YAML::Node& params) {
    auto opts = neuron::LIFOptions();
    if (params["v_rest"])   opts.v_rest(params["v_rest"].as<double>());
    if (params["v_thresh"]) opts.v_thresh(params["v_thresh"].as<double>());
    if (params["v_reset"])  opts.v_reset(params["v_reset"].as<double>());
    if (params["tau_m"])    opts.tau_m(params["tau_m"].as<double>());
    if (params["tau_ref"])  opts.tau_ref(params["tau_ref"].as<double>());
    if (params["c_m"])      opts.c_m(params["c_m"].as<double>());
    if (params["i_bg"])     opts.i_bg(params["i_bg"].as<double>());
    return opts;
}

static neuron::AdExOptions parse_adex_options(const YAML::Node& params) {
    auto opts = neuron::AdExOptions();
    if (params["v_rest"])   opts.v_rest(params["v_rest"].as<double>());
    if (params["v_thresh"]) opts.v_thresh(params["v_thresh"].as<double>());
    if (params["v_reset"])  opts.v_reset(params["v_reset"].as<double>());
    if (params["v_peak"])   opts.v_peak(params["v_peak"].as<double>());
    if (params["tau_m"])    opts.tau_m(params["tau_m"].as<double>());
    if (params["tau_w"])    opts.tau_w(params["tau_w"].as<double>());
    if (params["tau_ref"])  opts.tau_ref(params["tau_ref"].as<double>());
    if (params["c_m"])      opts.c_m(params["c_m"].as<double>());
    if (params["g_l"])      opts.g_l(params["g_l"].as<double>());
    if (params["a"])        opts.a(params["a"].as<double>());
    if (params["b"])        opts.b(params["b"].as<double>());
    if (params["delta_t"])  opts.delta_t(params["delta_t"].as<double>());
    if (params["i_bg"])     opts.i_bg(params["i_bg"].as<double>());
    return opts;
}

static neuron::IzhikevichOptions parse_izh_options(const YAML::Node& params) {
    auto opts = neuron::IzhikevichOptions();
    if (params["a"])      opts.a(params["a"].as<double>());
    if (params["b"])      opts.b(params["b"].as<double>());
    if (params["c"])      opts.c(params["c"].as<double>());
    if (params["d"])      opts.d(params["d"].as<double>());
    if (params["v_peak"]) opts.v_peak(params["v_peak"].as<double>());
    if (params["v_init"]) opts.v_init(params["v_init"].as<double>());
    if (params["u_init"]) opts.u_init(params["u_init"].as<double>());
    return opts;
}

static STDPOptions parse_stdp_options(const YAML::Node& params) {
    auto opts = STDPOptions();
    if (params["tau_plus"])      opts.tau_plus(params["tau_plus"].as<double>());
    if (params["tau_minus"])     opts.tau_minus(params["tau_minus"].as<double>());
    if (params["a_plus"])        opts.a_plus(params["a_plus"].as<double>());
    if (params["a_minus"])       opts.a_minus(params["a_minus"].as<double>());
    if (params["w_max"])         opts.w_max(params["w_max"].as<double>());
    if (params["w_min"])         opts.w_min(params["w_min"].as<double>());
    if (params["learning_rate"]) opts.learning_rate(params["learning_rate"].as<double>());
    return opts;
}

// ---------------------------------------------------------------------------
// Internal structures for holding created objects during config loading
// (so we can clean up on error and avoid leaks).
// ---------------------------------------------------------------------------

struct ConfigContext {
    // Neuron model pointers (owned, need destroy on error).
    std::vector<std::pair<std::string, void*>> neurons; // (model_type, ptr)
    // Topology generators (owned).
    std::vector<RandomTopology*> random_topos;
    std::vector<DistanceDepTopology*> dist_topos;
    // Synapse models (owned).
    std::vector<synapse::StaticSynapse*> synapses;
    // STDP states (owned).
    std::vector<STDPState*> stdp_states;

    // Populated populations by name.
    std::unordered_map<std::string, std::shared_ptr<Population>> populations;

    ~ConfigContext() {
        // Only destroy neurons — populations, connections, region are handed off.
        // Neuron model structs and topology/synapse/stdp are owned by the caller
        // or leaked. For simplicity we don't clean up here since load() either
        // succeeds (transfers ownership) or throws (best effort).
    }
};

// ---------------------------------------------------------------------------
// load_region()
// ---------------------------------------------------------------------------

Region* load_region(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    ConfigContext ctx;
    auto* region = region_create(
        root["region_name"] ? root["region_name"].as<std::string>() : "default");

    // --- Parse populations ---
    TORCH_CHECK(root["populations"], "Config must have 'populations' section");
    for (const auto& pop_node : root["populations"]) {
        auto name = pop_node["name"].as<std::string>();
        auto model = pop_node["model"].as<std::string>();
        auto count = pop_node["count"].as<int64_t>();
        auto params = pop_node["params"];

        NrnModule module{};
        if (model == "LIF" || model == "lif") {
            auto opts = params ? parse_lif_options(params) : neuron::LIFOptions();
            auto* lif = neuron::lif_create(count, opts);
            ctx.neurons.emplace_back("lif", lif);
            module = neuron::lif_as_module(lif);
        } else if (model == "AdEx" || model == "adex") {
            auto opts = params ? parse_adex_options(params) : neuron::AdExOptions();
            auto* adex = neuron::adex_create(count, opts);
            ctx.neurons.emplace_back("adex", adex);
            module = neuron::adex_as_module(adex);
        } else if (model == "Izhikevich" || model == "izhikevich" || model == "izh") {
            auto opts = params ? parse_izh_options(params) : neuron::IzhikevichOptions();
            auto* izh = neuron::izh_create(count, opts);
            ctx.neurons.emplace_back("izh", izh);
            module = neuron::izh_as_module(izh);
        } else {
            TORCH_CHECK(false, "Unknown neuron model: '", model, "'");
        }

        auto pop = std::shared_ptr<Population>(
            population_create(name, module, count, torch::kCPU),
            population_destroy);
        ctx.populations[name] = pop;
        region_add_population(region, pop);
    }

    // --- Parse connections ---
    if (root["connections"]) {
        for (const auto& conn_node : root["connections"]) {
            auto src_name = conn_node["source"].as<std::string>();
            auto tgt_name = conn_node["target"].as<std::string>();

            TORCH_CHECK(ctx.populations.count(src_name),
                        "Connection source '", src_name, "' not found");
            TORCH_CHECK(ctx.populations.count(tgt_name),
                        "Connection target '", tgt_name, "' not found");

            auto src = ctx.populations[src_name];
            auto tgt = ctx.populations[tgt_name];

            // Topology.
            auto topo_type = conn_node["topology"]
                ? conn_node["topology"].as<std::string>() : std::string("random");
            auto topo_params = conn_node["topology_params"];

            TopologyGenerator gen{};
            if (topo_type == "random") {
                auto ropts = RandomTopologyOptions();
                if (topo_params && topo_params["probability"])
                    ropts.probability(topo_params["probability"].as<double>());
                if (topo_params && topo_params["allow_autapses"])
                    ropts.allow_autapses(topo_params["allow_autapses"].as<bool>());
                auto* rt = random_topology_create(ropts);
                ctx.random_topos.push_back(rt);
                gen = random_topology_as_generator(rt);
            } else if (topo_type == "distance_dependent" || topo_type == "distance") {
                auto dopts = DistanceDependentOptions();
                if (topo_params && topo_params["sigma"])
                    dopts.sigma(topo_params["sigma"].as<double>());
                if (topo_params && topo_params["max_distance"])
                    dopts.max_distance(topo_params["max_distance"].as<double>());
                auto* dt = distance_dep_topology_create(dopts);
                ctx.dist_topos.push_back(dt);
                gen = distance_dep_topology_as_generator(dt);
            } else {
                TORCH_CHECK(false, "Unknown topology: '", topo_type, "'");
            }

            // Synapse.
            auto* syn = synapse::static_synapse_create(1);
            ctx.synapses.push_back(syn);
            auto syn_mod = synapse::static_synapse_as_module(syn);

            // Connection options.
            auto co = ConnectOptions();
            if (conn_node["block_size"])
                co.block_size(conn_node["block_size"].as<int64_t>());
            if (conn_node["weight_scale"])
                co.weight_scale(conn_node["weight_scale"].as<double>());
            if (conn_node["default_delay"])
                co.default_delay(conn_node["default_delay"].as<double>());

            auto conn = connect(src, tgt, &gen, syn_mod, co);

            // Plasticity.
            if (conn_node["plasticity"]) {
                auto plast_type = conn_node["plasticity"].as<std::string>();
                if (plast_type == "stdp" || plast_type == "STDP") {
                    auto popts = conn_node["plasticity_params"]
                        ? parse_stdp_options(conn_node["plasticity_params"])
                        : STDPOptions();
                    auto* stdp = stdp_create(popts);
                    ctx.stdp_states.push_back(stdp);
                    connection_attach(conn.get(), stdp_as_rule(stdp));
                }
            }

            region_add_connection(region, conn);
        }
    }

    return region;
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

Simulation* load(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);

    // Parse simulation options.
    auto sim_opts = SimulationOptions();
    if (root["simulation"]) {
        auto sn = root["simulation"];
        if (sn["duration"])       sim_opts.duration(sn["duration"].as<double>());
        if (sn["dt_fast"])        sim_opts.dt_fast(sn["dt_fast"].as<double>());
        if (sn["dt_slow"])        sim_opts.dt_slow(sn["dt_slow"].as<double>());
        if (sn["dt_structural"])  sim_opts.dt_structural(sn["dt_structural"].as<double>());
        if (sn["device"]) {
            auto dev_str = sn["device"].as<std::string>();
            if (dev_str == "cuda" || dev_str == "CUDA" || dev_str == "gpu") {
                sim_opts.device(torch::kCUDA);
            } else {
                sim_opts.device(torch::kCPU);
            }
        }
    }

    auto* region = load_region(path);
    return sim_create(region, sim_opts);
}

} // namespace config
} // namespace nrn
