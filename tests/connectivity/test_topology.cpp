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
    auto* gen = random_topology_create();
    SUCCEED();
    random_topology_destroy(gen);
}

TEST(RandomTopology, WithOptions) {
    RandomTopologyOptions opts;
    opts.probability(0.05);
    opts.allow_autapses(true);

    auto* gen = random_topology_create(opts);
    SUCCEED();
    random_topology_destroy(gen);
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
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.1));

    auto ct = random_topology_generate(
        static_cast<void*>(gen),
        /*n_source=*/64,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    // Basic shape checks on the result.
    EXPECT_EQ(ct.n_source, 64);
    EXPECT_EQ(ct.n_target, 64);
    EXPECT_EQ(ct.block_size, 32);
    EXPECT_GT(ct.block_index.n_blocks(), 0);

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateWeightsTensorExists) {
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.5));

    auto ct = random_topology_generate(
        static_cast<void*>(gen), 32, 32, 16, torch::kCPU);

    EXPECT_TRUE(ct.weights.defined());
    EXPECT_EQ(ct.weights.dim(), 3);
    EXPECT_EQ(ct.weights.size(1), 16);
    EXPECT_EQ(ct.weights.size(2), 16);

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateStructuralMaskExists) {
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.5));

    auto ct = random_topology_generate(
        static_cast<void*>(gen), 32, 32, 16, torch::kCPU);

    EXPECT_TRUE(ct.structural_mask.defined());
    EXPECT_EQ(ct.structural_mask.dim(), 3);
    EXPECT_EQ(ct.structural_mask.size(0), ct.block_index.n_blocks());

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateModulatoryMaskIsOnes) {
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.5));

    auto ct = random_topology_generate(
        static_cast<void*>(gen), 16, 16, 8, torch::kCPU);

    // Modulatory mask should be initialized to ones (no modulation).
    EXPECT_TRUE(ct.modulatory_mask.defined());
    EXPECT_TRUE(torch::all(ct.modulatory_mask == 1.0).item<bool>());

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateDelayTensorExists) {
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.5));

    auto ct = random_topology_generate(
        static_cast<void*>(gen), 16, 16, 8, torch::kCPU);

    EXPECT_TRUE(ct.delays.defined());
    EXPECT_EQ(ct.delays.dim(), 3);

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateBlockIndexConsistency) {
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.3));

    auto ct = random_topology_generate(
        static_cast<void*>(gen), 64, 64, 16, torch::kCPU);

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

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateDifferentSizes) {
    // Source and target can have different sizes.
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.2));

    auto ct = random_topology_generate(
        static_cast<void*>(gen),
        /*n_source=*/128,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 128);
    EXPECT_EQ(ct.n_target, 64);
    EXPECT_EQ(ct.block_size, 32);

    random_topology_destroy(gen);
}

TEST(RandomTopology, GenerateSmallPopulation) {
    // Population smaller than block size.
    auto* gen = random_topology_create(RandomTopologyOptions().probability(0.5));

    auto ct = random_topology_generate(
        static_cast<void*>(gen),
        /*n_source=*/8,
        /*n_target=*/8,
        /*block_size=*/16,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 8);
    EXPECT_EQ(ct.n_target, 8);
    EXPECT_EQ(ct.block_size, 16);

    random_topology_destroy(gen);
}

// ===========================================================================
// Distance-dependent topology generator
// ===========================================================================

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(DistanceDependent, DefaultConstruction) {
    auto* gen = distance_dep_topology_create();
    SUCCEED();
    distance_dep_topology_destroy(gen);
}

TEST(DistanceDependent, WithOptions) {
    auto opts = DistanceDependentOptions()
        .sigma(200.0_um)
        .max_distance(1000.0_um)
        .min_probability(0.01);

    auto* gen = distance_dep_topology_create(opts);
    SUCCEED();
    distance_dep_topology_destroy(gen);
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

    auto* gen = distance_dep_topology_create(opts);

    auto ct = distance_dep_topology_generate(
        static_cast<void*>(gen),
        /*n_source=*/64,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 64);
    EXPECT_EQ(ct.n_target, 64);
    EXPECT_EQ(ct.block_size, 32);

    distance_dep_topology_destroy(gen);
}

TEST(DistanceDependent, GenerateHasWeights) {
    auto* gen = distance_dep_topology_create(
        DistanceDependentOptions().sigma(100.0_um));

    auto ct = distance_dep_topology_generate(
        static_cast<void*>(gen), 32, 32, 16, torch::kCPU);

    EXPECT_TRUE(ct.weights.defined());
    EXPECT_TRUE(ct.structural_mask.defined());
    EXPECT_TRUE(ct.modulatory_mask.defined());
    EXPECT_TRUE(ct.delays.defined());

    distance_dep_topology_destroy(gen);
}

TEST(DistanceDependent, GenerateDifferentSizes) {
    auto* gen = distance_dep_topology_create();

    auto ct = distance_dep_topology_generate(
        static_cast<void*>(gen),
        /*n_source=*/128,
        /*n_target=*/64,
        /*block_size=*/32,
        /*device=*/torch::kCPU);

    EXPECT_EQ(ct.n_source, 128);
    EXPECT_EQ(ct.n_target, 64);

    distance_dep_topology_destroy(gen);
}

TEST(DistanceDependent, GenerateNarrowSigma) {
    // Very small sigma should produce sparser connectivity (fewer distant blocks).
    auto opts_narrow = DistanceDependentOptions()
        .sigma(10.0_um)
        .max_distance(50.0_um);

    auto* gen_narrow = distance_dep_topology_create(opts_narrow);

    auto ct_narrow = distance_dep_topology_generate(
        static_cast<void*>(gen_narrow), 64, 64, 16, torch::kCPU);

    // With very narrow sigma, we expect connectivity to be quite sparse.
    // Just verify it produced a valid structure.
    EXPECT_GE(ct_narrow.block_index.n_blocks(), 0);
    EXPECT_EQ(ct_narrow.block_size, 16);

    distance_dep_topology_destroy(gen_narrow);
}

TEST(DistanceDependent, GenerateWideSigma) {
    // Very wide sigma should produce denser connectivity.
    auto opts_wide = DistanceDependentOptions()
        .sigma(10000.0_um)
        .max_distance(100000.0_um);

    auto* gen_wide = distance_dep_topology_create(opts_wide);

    auto ct_wide = distance_dep_topology_generate(
        static_cast<void*>(gen_wide), 32, 32, 16, torch::kCPU);

    EXPECT_GT(ct_wide.block_index.n_blocks(), 0);

    distance_dep_topology_destroy(gen_wide);
}

// ===========================================================================
// TopologyGenerator polymorphism (ops-table dispatch)
// ===========================================================================

TEST(Topology, PolymorphicUsage) {
    // Both generators should be usable through the TopologyGenerator handle.
    auto* r = random_topology_create(RandomTopologyOptions().probability(0.1));
    auto gen1 = random_topology_as_generator(r);
    auto ct1 = topology_generate(&gen1, 32, 32, 16, torch::kCPU);
    EXPECT_EQ(ct1.n_source, 32);
    random_topology_destroy(r);

    auto* d = distance_dep_topology_create(DistanceDependentOptions().sigma(200.0_um));
    auto gen2 = distance_dep_topology_as_generator(d);
    auto ct2 = topology_generate(&gen2, 32, 32, 16, torch::kCPU);
    EXPECT_EQ(ct2.n_source, 32);
    distance_dep_topology_destroy(d);
}
