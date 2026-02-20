#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/connectivity/topology/random.h>
#include <nrn/connectivity/topology/distance_dependent.h>

using namespace nrn;
using namespace nrn::literals;

// ===========================================================================
// Random topology generator
// ===========================================================================

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(RandomTopology, DefaultConstruction) {
    // Constructing a Random topology generator should not crash.
    Random gen;
    SUCCEED();
}

TEST(RandomTopology, WithOptions) {
    RandomTopologyOptions opts;
    opts.probability(0.05);
    opts.allow_autapses(true);

    Random gen(opts);
    SUCCEED();
}

TEST(RandomTopology, OptionsBuilderChaining) {
    auto opts = RandomTopologyOptions()
        .probability(0.2)
        .allow_autapses(false);

    EXPECT_DOUBLE_EQ(opts.probability(), 0.2);
    EXPECT_FALSE(opts.allow_autapses());
}

TEST(RandomTopology, DefaultOptionsValues) {
    RandomTopologyOptions opts;
    EXPECT_DOUBLE_EQ(opts.probability(), 0.1);
    EXPECT_FALSE(opts.allow_autapses());
}

// ---------------------------------------------------------------------------
// generate produces valid structure
// ---------------------------------------------------------------------------

TEST(RandomTopology, GenerateBasicShape) {
    Random gen(RandomTopologyOptions().probability(0.1));

    auto ct = gen.generate(
        /*n_source=*/64,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    // Basic shape checks on the result.
    EXPECT_EQ(ct.n_source, 64);
    EXPECT_EQ(ct.n_target, 64);
    EXPECT_EQ(ct.block_size, 32);
    EXPECT_GT(ct.block_index.n_blocks(), 0);
}

TEST(RandomTopology, GenerateWeightsTensorExists) {
    Random gen(RandomTopologyOptions().probability(0.5));

    auto ct = gen.generate(32, 32, 16, torch::kCPU);

    EXPECT_TRUE(ct.weights.defined());
    EXPECT_EQ(ct.weights.dim(), 3);
    EXPECT_EQ(ct.weights.size(1), 16);
    EXPECT_EQ(ct.weights.size(2), 16);
}

TEST(RandomTopology, GenerateStructuralMaskExists) {
    Random gen(RandomTopologyOptions().probability(0.5));

    auto ct = gen.generate(32, 32, 16, torch::kCPU);

    EXPECT_TRUE(ct.structural_mask.defined());
    EXPECT_EQ(ct.structural_mask.dim(), 3);
    EXPECT_EQ(ct.structural_mask.size(0), ct.block_index.n_blocks());
}

TEST(RandomTopology, GenerateModulatoryMaskIsOnes) {
    Random gen(RandomTopologyOptions().probability(0.5));

    auto ct = gen.generate(16, 16, 8, torch::kCPU);

    // Modulatory mask should be initialized to ones (no modulation).
    EXPECT_TRUE(ct.modulatory_mask.defined());
    EXPECT_TRUE(torch::all(ct.modulatory_mask == 1.0).item<bool>());
}

TEST(RandomTopology, GenerateDelayTensorExists) {
    Random gen(RandomTopologyOptions().probability(0.5));

    auto ct = gen.generate(16, 16, 8, torch::kCPU);

    EXPECT_TRUE(ct.delays.defined());
    EXPECT_EQ(ct.delays.dim(), 3);
}

TEST(RandomTopology, GenerateBlockIndexConsistency) {
    Random gen(RandomTopologyOptions().probability(0.3));

    auto ct = gen.generate(64, 64, 16, torch::kCPU);

    int64_t n_blocks = ct.block_index.n_blocks();
    int64_t n_target_blocks = (64 + 16 - 1) / 16;  // = 4

    // row_ptr should have n_target_blocks + 1 entries.
    EXPECT_EQ(ct.block_index.row_ptr.size(0), n_target_blocks + 1);

    // The last entry in row_ptr should equal n_blocks.
    int32_t last_ptr = ct.block_index.row_ptr[-1].item<int32_t>();
    EXPECT_EQ(last_ptr, static_cast<int32_t>(n_blocks));

    // row_ptr should be non-decreasing.
    for (int64_t i = 1; i < ct.block_index.row_ptr.size(0); ++i) {
        EXPECT_GE(ct.block_index.row_ptr[i].item<int32_t>(),
                  ct.block_index.row_ptr[i-1].item<int32_t>());
    }
}

TEST(RandomTopology, GenerateDifferentSizes) {
    // Source and target can have different sizes.
    Random gen(RandomTopologyOptions().probability(0.2));

    auto ct = gen.generate(
        /*n_source=*/128,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 128);
    EXPECT_EQ(ct.n_target, 64);
    EXPECT_EQ(ct.block_size, 32);
}

TEST(RandomTopology, GenerateSmallPopulation) {
    // Population smaller than block size.
    Random gen(RandomTopologyOptions().probability(0.5));

    auto ct = gen.generate(
        /*n_source=*/8,
        /*n_target=*/8,
        /*block_size=*/16,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 8);
    EXPECT_EQ(ct.n_target, 8);
    EXPECT_EQ(ct.block_size, 16);
}

// ===========================================================================
// Distance-dependent topology generator
// ===========================================================================

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(DistanceDependent, DefaultConstruction) {
    DistanceDependent gen;
    SUCCEED();
}

TEST(DistanceDependent, WithOptions) {
    auto opts = DistanceDependentOptions()
        .sigma(200.0_um)
        .max_distance(1000.0_um)
        .min_probability(0.01);

    DistanceDependent gen(opts);
    SUCCEED();
}

TEST(DistanceDependent, DefaultOptionsValues) {
    DistanceDependentOptions opts;
    EXPECT_DOUBLE_EQ(opts.sigma(), 200.0e-6);           // 200 um
    EXPECT_DOUBLE_EQ(opts.max_distance(), 1000.0e-6);   // 1000 um
    EXPECT_DOUBLE_EQ(opts.min_probability(), 0.0);
}

TEST(DistanceDependent, OptionsBuilderChaining) {
    auto opts = DistanceDependentOptions()
        .sigma(150.0e-6)
        .max_distance(500.0e-6)
        .min_probability(0.05);

    EXPECT_DOUBLE_EQ(opts.sigma(), 150.0e-6);
    EXPECT_DOUBLE_EQ(opts.max_distance(), 500.0e-6);
    EXPECT_DOUBLE_EQ(opts.min_probability(), 0.05);
}

// ---------------------------------------------------------------------------
// generate produces valid structure
// ---------------------------------------------------------------------------

TEST(DistanceDependent, GenerateBasicShape) {
    auto opts = DistanceDependentOptions()
        .sigma(200.0_um)
        .max_distance(1000.0_um);

    DistanceDependent gen(opts);

    auto ct = gen.generate(
        /*n_source=*/64,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 64);
    EXPECT_EQ(ct.n_target, 64);
    EXPECT_EQ(ct.block_size, 32);
}

TEST(DistanceDependent, GenerateHasWeights) {
    DistanceDependent gen(DistanceDependentOptions().sigma(100.0_um));

    auto ct = gen.generate(32, 32, 16, torch::kCPU);

    EXPECT_TRUE(ct.weights.defined());
    EXPECT_TRUE(ct.structural_mask.defined());
    EXPECT_TRUE(ct.modulatory_mask.defined());
    EXPECT_TRUE(ct.delays.defined());
}

TEST(DistanceDependent, GenerateDifferentSizes) {
    DistanceDependent gen;

    auto ct = gen.generate(
        /*n_source=*/128,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 128);
    EXPECT_EQ(ct.n_target, 64);
}

TEST(DistanceDependent, GenerateNarrowSigma) {
    // Very small sigma should produce sparser connectivity (fewer distant blocks).
    auto opts_narrow = DistanceDependentOptions()
        .sigma(10.0_um)
        .max_distance(50.0_um);

    DistanceDependent gen_narrow(opts_narrow);

    auto ct_narrow = gen_narrow.generate(64, 64, 16, torch::kCPU);

    // With very narrow sigma, we expect connectivity to be quite sparse.
    // Just verify it produced a valid structure.
    EXPECT_GE(ct_narrow.block_index.n_blocks(), 0);
    EXPECT_EQ(ct_narrow.block_size, 16);
}

TEST(DistanceDependent, GenerateWideSigma) {
    // Very wide sigma should produce denser connectivity.
    auto opts_wide = DistanceDependentOptions()
        .sigma(10000.0_um)
        .max_distance(100000.0_um);

    DistanceDependent gen_wide(opts_wide);

    auto ct_wide = gen_wide.generate(32, 32, 16, torch::kCPU);

    EXPECT_GT(ct_wide.block_index.n_blocks(), 0);
}

// ===========================================================================
// TopologyGenerator polymorphism
// ===========================================================================

TEST(Topology, PolymorphicUsage) {
    // Both generators should be usable through the TopologyGenerator interface.
    std::unique_ptr<TopologyGenerator> gen;

    gen = std::make_unique<Random>(RandomTopologyOptions().probability(0.1));
    auto ct1 = gen->generate(32, 32, 16, torch::kCPU);
    EXPECT_EQ(ct1.n_source, 32);

    gen = std::make_unique<DistanceDependent>(DistanceDependentOptions().sigma(200.0_um));
    auto ct2 = gen->generate(32, 32, 16, torch::kCPU);
    EXPECT_EQ(ct2.n_source, 32);
}
