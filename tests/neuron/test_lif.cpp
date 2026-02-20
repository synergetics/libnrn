#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/neuron/lif.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(LIF, DefaultConstruction) {
    // Creating a LIF population should not crash.
    nrn::neuron::LIF lif(100);
    EXPECT_EQ(lif->size(), 100);
}

TEST(LIF, ConstructionWithOptions) {
    auto opts = neuron::LIFOptions()
        .v_rest(-0.070)
        .v_thresh(-0.050)
        .tau_m(0.020);

    nrn::neuron::LIF lif(50, opts);
    EXPECT_EQ(lif->size(), 50);
}

// ---------------------------------------------------------------------------
// Reset sets v to v_rest
// ---------------------------------------------------------------------------

TEST(LIF, ResetSetsVToVRest) {
    nrn::neuron::LIF lif(100);

    // After construction, v should be at v_rest.
    auto v = lif->v;
    auto v_rest = lif->v_rest;

    EXPECT_EQ(v.size(0), 100);
    EXPECT_EQ(v_rest.size(0), 100);

    // All v values should equal v_rest.
    EXPECT_TRUE(torch::allclose(v, v_rest));
}

TEST(LIF, ResetClearsSpikes) {
    nrn::neuron::LIF lif(100);

    // Spikes should be zero after reset.
    auto spikes = lif->spike;
    EXPECT_TRUE(torch::all(spikes == 0).item<bool>());
}

// ---------------------------------------------------------------------------
// State variable names
// ---------------------------------------------------------------------------

TEST(LIF, StateVars) {
    nrn::neuron::LIF lif(10);
    auto vars = lif->state_vars();
    // Should include at least "v" and "spike".
    bool has_v = false, has_spike = false;
    for (const auto& var : vars) {
        if (var == "v") has_v = true;
        if (var == "spike") has_spike = true;
    }
    EXPECT_TRUE(has_v);
    EXPECT_TRUE(has_spike);
}
