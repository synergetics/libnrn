#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/neuron/adex.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(AdEx, DefaultConstruction) {
    // Creating an AdEx population should not crash.
    nrn::neuron::AdEx adex(100);
    EXPECT_EQ(adex->size(), 100);
}

TEST(AdEx, ConstructionWithOptions) {
    auto opts = neuron::AdExOptions()
        .v_rest(-70.6_mV)
        .v_thresh(-50.4_mV)
        .tau_m(9.37_ms)
        .a(4.0_nS)
        .b(80.5e-12)
        .delta_t(2.0_mV);

    nrn::neuron::AdEx adex(80000, opts);
    EXPECT_EQ(adex->size(), 80000);
}

TEST(AdEx, SingleNeuron) {
    nrn::neuron::AdEx adex(1);
    EXPECT_EQ(adex->size(), 1);
    EXPECT_EQ(adex->v.size(0), 1);
    EXPECT_EQ(adex->w.size(0), 1);
}

// ---------------------------------------------------------------------------
// State tensors exist and have correct shape
// ---------------------------------------------------------------------------

TEST(AdEx, StateTensorsShape) {
    nrn::neuron::AdEx adex(200);

    EXPECT_EQ(adex->v.size(0), 200);
    EXPECT_EQ(adex->w.size(0), 200);
    EXPECT_EQ(adex->spike.size(0), 200);
    EXPECT_EQ(adex->refractory.size(0), 200);
    EXPECT_EQ(adex->I_syn.size(0), 200);
}

TEST(AdEx, ParameterTensorsShape) {
    nrn::neuron::AdEx adex(64);

    EXPECT_EQ(adex->v_rest.size(0), 64);
    EXPECT_EQ(adex->v_thresh.size(0), 64);
    EXPECT_EQ(adex->v_reset.size(0), 64);
    EXPECT_EQ(adex->v_peak.size(0), 64);
    EXPECT_EQ(adex->tau_m.size(0), 64);
    EXPECT_EQ(adex->tau_w.size(0), 64);
    EXPECT_EQ(adex->tau_ref.size(0), 64);
    EXPECT_EQ(adex->c_m.size(0), 64);
    EXPECT_EQ(adex->g_l.size(0), 64);
    EXPECT_EQ(adex->a.size(0), 64);
    EXPECT_EQ(adex->b.size(0), 64);
    EXPECT_EQ(adex->delta_t.size(0), 64);
    EXPECT_EQ(adex->i_bg.size(0), 64);
}

// ---------------------------------------------------------------------------
// Reset sets v to v_rest
// ---------------------------------------------------------------------------

TEST(AdEx, ResetSetsVToVRest) {
    nrn::neuron::AdEx adex(100);

    // After construction, v should be at v_rest.
    auto v = adex->v;
    auto v_rest = adex->v_rest;

    EXPECT_EQ(v.size(0), 100);
    EXPECT_EQ(v_rest.size(0), 100);

    // All v values should equal v_rest.
    EXPECT_TRUE(torch::allclose(v, v_rest));
}

TEST(AdEx, ResetClearsSpikes) {
    nrn::neuron::AdEx adex(100);

    // Spikes should be zero after reset.
    auto spikes = adex->spike;
    EXPECT_TRUE(torch::all(spikes == 0).item<bool>());
}

TEST(AdEx, ResetClearsAdaptation) {
    nrn::neuron::AdEx adex(50);

    // Adaptation current w should be zero after reset.
    auto w = adex->w;
    EXPECT_TRUE(torch::all(w == 0).item<bool>());
}

TEST(AdEx, ResetClearsRefractory) {
    nrn::neuron::AdEx adex(50);

    // Refractory timer should be zero after reset.
    auto ref = adex->refractory;
    EXPECT_TRUE(torch::all(ref == 0).item<bool>());
}

TEST(AdEx, ResetClearsSynapticCurrent) {
    nrn::neuron::AdEx adex(50);

    // I_syn should be zero after reset.
    auto isyn = adex->I_syn;
    EXPECT_TRUE(torch::all(isyn == 0).item<bool>());
}

TEST(AdEx, ExplicitResetRestoresState) {
    nrn::neuron::AdEx adex(32);

    // Manually perturb state tensors.
    adex->v.fill_(0.0);
    adex->w.fill_(999.0);
    adex->spike.fill_(1.0);

    // Reset should restore initial conditions.
    adex->reset();

    EXPECT_TRUE(torch::allclose(adex->v, adex->v_rest));
    EXPECT_TRUE(torch::all(adex->w == 0).item<bool>());
    EXPECT_TRUE(torch::all(adex->spike == 0).item<bool>());
}

// ---------------------------------------------------------------------------
// Custom options propagate to parameter tensors
// ---------------------------------------------------------------------------

TEST(AdEx, CustomOptionsPropagate) {
    double custom_v_rest = -75.0_mV;
    auto opts = neuron::AdExOptions().v_rest(custom_v_rest);

    nrn::neuron::AdEx adex(10, opts);

    // v_rest tensor should be filled with the custom value.
    auto v_rest_val = adex->v_rest[0].item<float>();
    EXPECT_NEAR(v_rest_val, static_cast<float>(custom_v_rest), 1e-6);

    // v should also be initialized to custom v_rest.
    auto v_val = adex->v[0].item<float>();
    EXPECT_NEAR(v_val, static_cast<float>(custom_v_rest), 1e-6);
}

// ---------------------------------------------------------------------------
// State variable names
// ---------------------------------------------------------------------------

TEST(AdEx, StateVars) {
    nrn::neuron::AdEx adex(10);
    auto vars = adex->state_vars();
    // Should include at least "v", "w", and "spike".
    bool has_v = false, has_w = false, has_spike = false;
    for (const auto& var : vars) {
        if (var == "v") has_v = true;
        if (var == "w") has_w = true;
        if (var == "spike") has_spike = true;
    }
    EXPECT_TRUE(has_v);
    EXPECT_TRUE(has_w);
    EXPECT_TRUE(has_spike);
}

TEST(AdEx, StateVarsIncludeISyn) {
    nrn::neuron::AdEx adex(10);
    auto vars = adex->state_vars();

    bool has_isyn = false;
    for (const auto& var : vars) {
        if (var == "I_syn") has_isyn = true;
    }
    EXPECT_TRUE(has_isyn);
}

TEST(AdEx, StateVarsIncludeRefractory) {
    nrn::neuron::AdEx adex(10);
    auto vars = adex->state_vars();

    bool has_ref = false;
    for (const auto& var : vars) {
        if (var == "refractory") has_ref = true;
    }
    EXPECT_TRUE(has_ref);
}

// ---------------------------------------------------------------------------
// Default options match Brette & Gerstner (2005) values
// ---------------------------------------------------------------------------

TEST(AdEx, DefaultOptionsMatchBretteGerstner) {
    neuron::AdExOptions opts;

    EXPECT_DOUBLE_EQ(opts.v_rest(), -0.0706);     // -70.6 mV
    EXPECT_DOUBLE_EQ(opts.v_thresh(), -0.0504);    // -50.4 mV
    EXPECT_DOUBLE_EQ(opts.v_peak(), 0.0);          // 0 mV
    EXPECT_DOUBLE_EQ(opts.tau_w(), 0.144);         // 144 ms
    EXPECT_DOUBLE_EQ(opts.c_m(), 281e-12);         // 281 pF
    EXPECT_DOUBLE_EQ(opts.g_l(), 30e-9);           // 30 nS
    EXPECT_DOUBLE_EQ(opts.a(), 4e-9);              // 4 nS
    EXPECT_DOUBLE_EQ(opts.b(), 80.5e-12);          // 0.0805 nA
    EXPECT_DOUBLE_EQ(opts.delta_t(), 0.002);       // 2 mV
}
