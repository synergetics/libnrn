#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/plasticity/stdp.h>
#include <nrn/plasticity/plasticity_rule.h>
#include <nrn/plasticity/options.h>
#include <nrn/connectivity/connectivity_tensor.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Helper: create a minimal ConnectivityTensor for STDP tests.
// ---------------------------------------------------------------------------

static ConnectivityTensor make_test_ct(int64_t n_blocks, int64_t block_size) {
    ConnectivityTensor ct;
    ct.block_size = block_size;
    ct.n_source = n_blocks * block_size;
    ct.n_target = n_blocks * block_size;

    int64_t B = block_size;

    std::vector<int32_t> row_ptr_vec(n_blocks + 1);
    std::vector<int32_t> col_idx_vec(n_blocks);
    for (int64_t i = 0; i <= n_blocks; ++i) {
        row_ptr_vec[i] = static_cast<int32_t>(i);
    }
    for (int64_t i = 0; i < n_blocks; ++i) {
        col_idx_vec[i] = static_cast<int32_t>(i);
    }

    ct.block_index.row_ptr = torch::tensor(row_ptr_vec, torch::kInt32);
    ct.block_index.col_idx = torch::tensor(col_idx_vec, torch::kInt32);

    ct.weights         = torch::full({n_blocks, B, B}, 0.5f);
    ct.structural_mask = torch::ones({n_blocks, B, B});
    ct.modulatory_mask = torch::ones({n_blocks, B, B});
    ct.delays          = torch::zeros({n_blocks, B, B}, torch::kInt32);

    return ct;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(STDP, DefaultConstruction) {
    // Constructing an STDP rule with default options should not crash.
    auto* stdp = stdp_create();
    SUCCEED();
    stdp_destroy(stdp);
}

TEST(STDP, ConstructionWithOptions) {
    auto opts = STDPOptions()
        .tau_plus(0.020)
        .tau_minus(0.020)
        .a_plus(0.01)
        .a_minus(-0.012)
        .w_max(1.0)
        .w_min(0.0)
        .learning_rate(0.5);

    auto* stdp = stdp_create(opts);

    EXPECT_DOUBLE_EQ(stdp_options(stdp).tau_plus(), 0.020);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).tau_minus(), 0.020);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).a_plus(), 0.01);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).a_minus(), -0.012);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).w_max(), 1.0);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).w_min(), 0.0);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).learning_rate(), 0.5);

    stdp_destroy(stdp);
}

// ---------------------------------------------------------------------------
// Default options values
// ---------------------------------------------------------------------------

TEST(STDP, DefaultOptionsValues) {
    STDPOptions opts;

    EXPECT_DOUBLE_EQ(opts.tau_plus(), 0.020);
    EXPECT_DOUBLE_EQ(opts.tau_minus(), 0.020);
    EXPECT_DOUBLE_EQ(opts.a_plus(), 0.01);
    EXPECT_DOUBLE_EQ(opts.a_minus(), -0.012);
    EXPECT_DOUBLE_EQ(opts.w_max(), 1.0);
    EXPECT_DOUBLE_EQ(opts.w_min(), 0.0);
    EXPECT_DOUBLE_EQ(opts.learning_rate(), 1.0);
}

// ---------------------------------------------------------------------------
// Options builder chaining
// ---------------------------------------------------------------------------

TEST(STDP, OptionsBuilderChaining) {
    auto opts = STDPOptions()
        .tau_plus(15.0_ms)
        .tau_minus(25.0_ms)
        .a_plus(0.005)
        .a_minus(-0.005)
        .w_max(2.0)
        .w_min(-0.5)
        .learning_rate(0.1);

    EXPECT_DOUBLE_EQ(opts.tau_plus(), 15.0e-3);
    EXPECT_DOUBLE_EQ(opts.tau_minus(), 25.0e-3);
    EXPECT_DOUBLE_EQ(opts.a_plus(), 0.005);
    EXPECT_DOUBLE_EQ(opts.a_minus(), -0.005);
    EXPECT_DOUBLE_EQ(opts.w_max(), 2.0);
    EXPECT_DOUBLE_EQ(opts.w_min(), -0.5);
    EXPECT_DOUBLE_EQ(opts.learning_rate(), 0.1);
}

// ---------------------------------------------------------------------------
// Reset does not crash
// ---------------------------------------------------------------------------

TEST(STDP, ResetDoesNotCrash) {
    auto* stdp = stdp_create();
    stdp_reset(stdp);
    SUCCEED();
    stdp_destroy(stdp);
}

// ---------------------------------------------------------------------------
// Initialize allocates trace tensors
// ---------------------------------------------------------------------------

TEST(STDP, InitializeAllocatesTraces) {
    auto* stdp = stdp_create();
    auto ct = make_test_ct(2, 4);

    // Before initialize, traces should not exist.
    EXPECT_FALSE(ct.trace_pre.has_value());
    EXPECT_FALSE(ct.trace_post.has_value());

    stdp_initialize(stdp, ct);

    // After initialize, traces should exist with correct shape.
    EXPECT_TRUE(ct.trace_pre.has_value());
    EXPECT_TRUE(ct.trace_post.has_value());
    EXPECT_EQ(ct.trace_pre->sizes(), ct.weights.sizes());
    EXPECT_EQ(ct.trace_post->sizes(), ct.weights.sizes());

    stdp_destroy(stdp);
}

TEST(STDP, InitializeTracesAreZero) {
    auto* stdp = stdp_create();
    auto ct = make_test_ct(2, 4);

    stdp_initialize(stdp, ct);

    EXPECT_TRUE(torch::all(*ct.trace_pre == 0).item<bool>());
    EXPECT_TRUE(torch::all(*ct.trace_post == 0).item<bool>());

    stdp_destroy(stdp);
}

// ---------------------------------------------------------------------------
// Polymorphic usage through PlasticityRule ops table
// ---------------------------------------------------------------------------

TEST(STDP, PolymorphicUsage) {
    auto* stdp = stdp_create(
        STDPOptions().tau_plus(20.0_ms).a_plus(0.01));

    auto rule = stdp_as_rule(stdp);

    auto ct = make_test_ct(1, 4);
    plasticity_initialize(&rule, ct);
    plasticity_reset(&rule);

    EXPECT_TRUE(ct.trace_pre.has_value());
    SUCCEED();

    stdp_destroy(stdp);
}

// ---------------------------------------------------------------------------
// STDP preserves weight bounds
// ---------------------------------------------------------------------------

TEST(STDP, WeightBoundsInOptions) {
    auto opts = STDPOptions().w_min(0.0).w_max(5.0);
    auto* stdp = stdp_create(opts);

    EXPECT_DOUBLE_EQ(stdp_options(stdp).w_min(), 0.0);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).w_max(), 5.0);

    stdp_destroy(stdp);
}

// ---------------------------------------------------------------------------
// STDP with asymmetric time constants
// ---------------------------------------------------------------------------

TEST(STDP, AsymmetricTimeConstants) {
    auto opts = STDPOptions()
        .tau_plus(10.0_ms)
        .tau_minus(30.0_ms);

    auto* stdp = stdp_create(opts);

    EXPECT_DOUBLE_EQ(stdp_options(stdp).tau_plus(), 10.0e-3);
    EXPECT_DOUBLE_EQ(stdp_options(stdp).tau_minus(), 30.0e-3);
    EXPECT_NE(stdp_options(stdp).tau_plus(), stdp_options(stdp).tau_minus());

    stdp_destroy(stdp);
}

// ---------------------------------------------------------------------------
// STDP with custom learning rate
// ---------------------------------------------------------------------------

TEST(STDP, ZeroLearningRate) {
    auto opts = STDPOptions().learning_rate(0.0);
    auto* stdp = stdp_create(opts);

    EXPECT_DOUBLE_EQ(stdp_options(stdp).learning_rate(), 0.0);

    stdp_destroy(stdp);
}

TEST(STDP, HighLearningRate) {
    auto opts = STDPOptions().learning_rate(10.0);
    auto* stdp = stdp_create(opts);

    EXPECT_DOUBLE_EQ(stdp_options(stdp).learning_rate(), 10.0);

    stdp_destroy(stdp);
}
