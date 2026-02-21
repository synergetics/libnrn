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
    auto* lif = neuron::lif_create(100);
    EXPECT_EQ(lif->n, 100);
    neuron::lif_destroy(lif);
}

TEST(LIF, ConstructionWithOptions) {
    auto opts = neuron::LIFOptions()
        .v_rest(-0.070)
        .v_thresh(-0.050)
        .tau_m(0.020);

    auto* lif = neuron::lif_create(50, opts);
    EXPECT_EQ(lif->n, 50);
    neuron::lif_destroy(lif);
}

// ---------------------------------------------------------------------------
// Reset sets v to v_rest
// ---------------------------------------------------------------------------

TEST(LIF, ResetSetsVToVRest) {
    auto* lif = neuron::lif_create(100);

    // After construction, v should be at v_rest.
    auto v = lif->v;
    auto v_rest = lif->v_rest;

    EXPECT_EQ(v.size(0), 100);
    EXPECT_EQ(v_rest.size(0), 100);

    // All v values should equal v_rest.
    EXPECT_TRUE(torch::allclose(v, v_rest));
    neuron::lif_destroy(lif);
}

TEST(LIF, ResetClearsSpikes) {
    auto* lif = neuron::lif_create(100);

    // Spikes should be zero after reset.
    auto spikes = lif->spike;
    EXPECT_TRUE(torch::all(spikes == 0).item<bool>());
    neuron::lif_destroy(lif);
}

// ---------------------------------------------------------------------------
// State variable names
// ---------------------------------------------------------------------------

TEST(LIF, StateVars) {
    auto* lif = neuron::lif_create(10);
    int count = 0;
    auto* vars = neuron::lif_state_vars(lif, &count);
    // Should include at least "v" and "spike".
    bool has_v = false, has_spike = false;
    for (int i = 0; i < count; ++i) {
        if (std::string(vars[i]) == "v") has_v = true;
        if (std::string(vars[i]) == "spike") has_spike = true;
    }
    EXPECT_TRUE(has_v);
    EXPECT_TRUE(has_spike);
    neuron::lif_destroy(lif);
}
