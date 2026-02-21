#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/neuron/izhikevich.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(Izhikevich, DefaultConstruction) {
    // Creating an Izhikevich population should not crash.
    auto* izh = neuron::izh_create(100);
    EXPECT_EQ(izh->n, 100);
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ConstructionWithOptions) {
    // Fast Spiking (FS) parameters.
    auto opts = neuron::IzhikevichOptions()
        .a(0.1)
        .b(0.2)
        .c(-65.0)
        .d(2.0)
        .v_peak(30.0);

    auto* izh = neuron::izh_create(50, opts);
    EXPECT_EQ(izh->n, 50);
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, SingleNeuron) {
    auto* izh = neuron::izh_create(1);
    EXPECT_EQ(izh->n, 1);
    EXPECT_EQ(izh->v.size(0), 1);
    EXPECT_EQ(izh->u.size(0), 1);
    neuron::izh_destroy(izh);
}

// ---------------------------------------------------------------------------
// Izhikevich neuron type presets
// ---------------------------------------------------------------------------

TEST(Izhikevich, RegularSpikingDefaults) {
    // Default options should correspond to Regular Spiking (RS).
    neuron::IzhikevichOptions opts;
    EXPECT_DOUBLE_EQ(opts.a(), 0.02);
    EXPECT_DOUBLE_EQ(opts.b(), 0.2);
    EXPECT_DOUBLE_EQ(opts.c(), -65.0);
    EXPECT_DOUBLE_EQ(opts.d(), 8.0);
    EXPECT_DOUBLE_EQ(opts.v_peak(), 30.0);
    EXPECT_DOUBLE_EQ(opts.v_init(), -65.0);
    EXPECT_DOUBLE_EQ(opts.u_init(), 0.0);
}

TEST(Izhikevich, FastSpikingOptions) {
    // FS neuron parameters from Izhikevich (2003).
    auto opts = neuron::IzhikevichOptions()
        .a(0.1)
        .b(0.2)
        .c(-65.0)
        .d(2.0);

    auto* izh = neuron::izh_create(100, opts);
    EXPECT_EQ(izh->n, 100);
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ChatteringOptions) {
    // CH neuron parameters from Izhikevich (2003).
    auto opts = neuron::IzhikevichOptions()
        .a(0.02)
        .b(0.2)
        .c(-50.0)
        .d(2.0);

    auto* izh = neuron::izh_create(100, opts);
    EXPECT_EQ(izh->n, 100);
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, IntrinsicallyBurstingOptions) {
    // IB neuron parameters from Izhikevich (2003).
    auto opts = neuron::IzhikevichOptions()
        .a(0.02)
        .b(0.2)
        .c(-55.0)
        .d(4.0);

    auto* izh = neuron::izh_create(100, opts);
    EXPECT_EQ(izh->n, 100);
    neuron::izh_destroy(izh);
}

// ---------------------------------------------------------------------------
// State tensors exist and have correct shape
// ---------------------------------------------------------------------------

TEST(Izhikevich, StateTensorsShape) {
    auto* izh = neuron::izh_create(256);

    EXPECT_EQ(izh->v.size(0), 256);
    EXPECT_EQ(izh->u.size(0), 256);
    EXPECT_EQ(izh->spike.size(0), 256);
    EXPECT_EQ(izh->I_syn.size(0), 256);
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ParameterTensorsShape) {
    auto* izh = neuron::izh_create(128);

    EXPECT_EQ(izh->a.size(0), 128);
    EXPECT_EQ(izh->b.size(0), 128);
    EXPECT_EQ(izh->c.size(0), 128);
    EXPECT_EQ(izh->d.size(0), 128);
    EXPECT_EQ(izh->v_peak.size(0), 128);
    neuron::izh_destroy(izh);
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

TEST(Izhikevich, ResetSetsVToVInit) {
    auto opts = neuron::IzhikevichOptions().v_init(-65.0);
    auto* izh = neuron::izh_create(100, opts);

    // After construction, v should be initialized to v_init.
    auto v = izh->v;
    EXPECT_EQ(v.size(0), 100);

    // All v values should be v_init.
    auto expected = torch::full({100}, -65.0f);
    EXPECT_TRUE(torch::allclose(v, expected));
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ResetClearsSpikes) {
    auto* izh = neuron::izh_create(100);

    // Spikes should be zero after reset.
    auto spikes = izh->spike;
    EXPECT_TRUE(torch::all(spikes == 0).item<bool>());
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ResetClearsSynapticCurrent) {
    auto* izh = neuron::izh_create(50);

    // I_syn should be zero after reset.
    auto isyn = izh->I_syn;
    EXPECT_TRUE(torch::all(isyn == 0).item<bool>());
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ExplicitResetRestoresState) {
    auto* izh = neuron::izh_create(32);

    // Manually perturb state tensors.
    izh->v.fill_(0.0);
    izh->u.fill_(999.0);
    izh->spike.fill_(1.0);

    // Reset should restore initial conditions.
    neuron::izh_reset(izh);

    EXPECT_TRUE(torch::all(izh->spike == 0).item<bool>());
    EXPECT_TRUE(torch::all(izh->I_syn == 0).item<bool>());
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, ResetWithCustomVInit) {
    auto opts = neuron::IzhikevichOptions().v_init(-70.0).u_init(5.0);
    auto* izh = neuron::izh_create(16, opts);

    // Perturb and reset.
    izh->v.fill_(100.0);
    izh->u.fill_(-100.0);
    neuron::izh_reset(izh);

    auto v_val = izh->v[0].item<float>();
    EXPECT_NEAR(v_val, -70.0f, 1e-5);
    neuron::izh_destroy(izh);
}

// ---------------------------------------------------------------------------
// State variable names
// ---------------------------------------------------------------------------

TEST(Izhikevich, StateVars) {
    auto* izh = neuron::izh_create(10);
    int count = 0;
    auto* vars = neuron::izh_state_vars(izh, &count);
    // Should include at least "v", "u", and "spike".
    bool has_v = false, has_u = false, has_spike = false;
    for (int i = 0; i < count; ++i) {
        if (std::string(vars[i]) == "v") has_v = true;
        if (std::string(vars[i]) == "u") has_u = true;
        if (std::string(vars[i]) == "spike") has_spike = true;
    }
    EXPECT_TRUE(has_v);
    EXPECT_TRUE(has_u);
    EXPECT_TRUE(has_spike);
    neuron::izh_destroy(izh);
}

TEST(Izhikevich, StateVarsIncludeISyn) {
    auto* izh = neuron::izh_create(10);
    int count = 0;
    auto* vars = neuron::izh_state_vars(izh, &count);

    bool has_isyn = false;
    for (int i = 0; i < count; ++i) {
        if (std::string(vars[i]) == "I_syn") has_isyn = true;
    }
    EXPECT_TRUE(has_isyn);
    neuron::izh_destroy(izh);
}

// ---------------------------------------------------------------------------
// Options builder pattern
// ---------------------------------------------------------------------------

TEST(Izhikevich, OptionsBuilderChaining) {
    // Verify fluent setter chaining works correctly.
    auto opts = neuron::IzhikevichOptions()
        .a(0.05)
        .b(0.25)
        .c(-50.0)
        .d(6.0)
        .v_peak(35.0)
        .v_init(-60.0)
        .u_init(1.0);

    EXPECT_DOUBLE_EQ(opts.a(), 0.05);
    EXPECT_DOUBLE_EQ(opts.b(), 0.25);
    EXPECT_DOUBLE_EQ(opts.c(), -50.0);
    EXPECT_DOUBLE_EQ(opts.d(), 6.0);
    EXPECT_DOUBLE_EQ(opts.v_peak(), 35.0);
    EXPECT_DOUBLE_EQ(opts.v_init(), -60.0);
    EXPECT_DOUBLE_EQ(opts.u_init(), 1.0);
}
