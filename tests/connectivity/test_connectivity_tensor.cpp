#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/connectivity/connectivity_tensor.h>

using namespace nrn;

// ---------------------------------------------------------------------------
// Helper: create a minimal ConnectivityTensor for testing.
// ---------------------------------------------------------------------------

static ConnectivityTensor make_small_ct(
    int64_t n_blocks, int64_t block_size,
    int64_t n_source, int64_t n_target,
    float structural_fill = 1.0f) {

    ConnectivityTensor ct;
    ct.block_size = block_size;
    ct.n_source = n_source;
    ct.n_target = n_target;

    int64_t B = block_size;
    int64_t n_target_blocks = (n_target + B - 1) / B;

    // Build a simple block index with n_blocks entries.
    // Distribute blocks across target rows (round-robin).
    std::vector<int32_t> row_ptr_vec(n_target_blocks + 1, 0);
    std::vector<int32_t> col_idx_vec;

    int64_t assigned = 0;
    for (int64_t row = 0; row < n_target_blocks && assigned < n_blocks; ++row) {
        col_idx_vec.push_back(static_cast<int32_t>(row % ((n_source + B - 1) / B)));
        assigned++;
        row_ptr_vec[row + 1] = static_cast<int32_t>(assigned);
    }
    // Fill remaining rows with same offset as last assigned.
    for (int64_t row = assigned; row < n_target_blocks; ++row) {
        row_ptr_vec[row + 1] = row_ptr_vec[row];
    }

    ct.block_index.row_ptr = torch::tensor(row_ptr_vec, torch::kInt32);
    ct.block_index.col_idx = torch::tensor(col_idx_vec, torch::kInt32);

    ct.weights         = torch::ones({n_blocks, B, B});
    ct.structural_mask = torch::full({n_blocks, B, B}, structural_fill);
    ct.modulatory_mask = torch::ones({n_blocks, B, B});
    ct.delays          = torch::zeros({n_blocks, B, B}, torch::kInt32);

    return ct;
}

// ---------------------------------------------------------------------------
// BlockIndex
// ---------------------------------------------------------------------------

TEST(BlockIndex, NBlocksAndNRows) {
    BlockIndex bi;
    bi.row_ptr = torch::tensor({0, 2, 3, 5}, torch::kInt32);
    bi.col_idx = torch::tensor({0, 1, 2, 0, 3}, torch::kInt32);

    EXPECT_EQ(bi.n_blocks(), 5);
    EXPECT_EQ(bi.n_rows(), 3);
}

TEST(BlockIndex, EmptyBlockIndex) {
    BlockIndex bi;
    bi.row_ptr = torch::tensor({0}, torch::kInt32);
    bi.col_idx = torch::zeros({0}, torch::kInt32);

    EXPECT_EQ(bi.n_blocks(), 0);
    EXPECT_EQ(bi.n_rows(), 0);
}

// ---------------------------------------------------------------------------
// Manual construction of a small ConnectivityTensor
// ---------------------------------------------------------------------------

TEST(ConnectivityTensor, ManualConstruction) {
    // Create a tiny connectivity: 2 blocks, block_size = 4.
    // This represents connectivity between 8 source and 8 target neurons,
    // with 2 active dense blocks.

    ConnectivityTensor ct;
    ct.block_size = 4;
    ct.n_source = 8;
    ct.n_target = 8;

    // Block index: 2 target block rows, each with 1 source column block.
    // Row 0 connects to source block 0; row 1 connects to source block 1.
    ct.block_index.row_ptr = torch::tensor({0, 1, 2}, torch::kInt32);
    ct.block_index.col_idx = torch::tensor({0, 1}, torch::kInt32);

    int64_t n_blocks = 2;
    int64_t B = ct.block_size;

    ct.weights         = torch::ones({n_blocks, B, B});
    ct.structural_mask = torch::ones({n_blocks, B, B});
    ct.modulatory_mask = torch::ones({n_blocks, B, B});
    ct.delays          = torch::zeros({n_blocks, B, B}, torch::kInt32);

    EXPECT_EQ(ct.block_index.n_blocks(), 2);
    EXPECT_EQ(ct.block_index.n_rows(), 2);
}

TEST(ConnectivityTensor, FieldDimensions) {
    auto ct = make_small_ct(4, 8, 32, 32);

    EXPECT_EQ(ct.block_size, 8);
    EXPECT_EQ(ct.n_source, 32);
    EXPECT_EQ(ct.n_target, 32);
    EXPECT_EQ(ct.weights.dim(), 3);
    EXPECT_EQ(ct.weights.size(1), 8);
    EXPECT_EQ(ct.weights.size(2), 8);
}

// ---------------------------------------------------------------------------
// effective_weights
// ---------------------------------------------------------------------------

TEST(ConnectivityTensor, EffectiveWeightsShape) {
    ConnectivityTensor ct;
    ct.block_size = 8;
    ct.n_source = 16;
    ct.n_target = 16;

    int64_t n_blocks = 4;
    int64_t B = ct.block_size;

    ct.block_index.row_ptr = torch::tensor({0, 2, 4}, torch::kInt32);
    ct.block_index.col_idx = torch::tensor({0, 1, 0, 1}, torch::kInt32);

    ct.weights         = torch::randn({n_blocks, B, B});
    ct.structural_mask = torch::ones({n_blocks, B, B});
    ct.modulatory_mask = torch::ones({n_blocks, B, B});
    ct.delays          = torch::zeros({n_blocks, B, B}, torch::kInt32);

    auto ew = ct.effective_weights();

    // effective_weights should have the same shape as weights.
    EXPECT_EQ(ew.sizes(), ct.weights.sizes());
    EXPECT_EQ(ew.size(0), n_blocks);
    EXPECT_EQ(ew.size(1), B);
    EXPECT_EQ(ew.size(2), B);
}

TEST(ConnectivityTensor, EffectiveWeightsAllOnes) {
    // When all masks are 1, effective_weights should equal weights.
    auto ct = make_small_ct(2, 4, 8, 8);
    ct.weights = torch::randn({2, 4, 4});

    auto ew = ct.effective_weights();
    EXPECT_TRUE(torch::allclose(ew, ct.weights));
}

TEST(ConnectivityTensor, EffectiveWeightsZeroStructuralMask) {
    // When structural_mask is zero, effective_weights should be all zeros.
    auto ct = make_small_ct(2, 4, 8, 8, /*structural_fill=*/0.0f);
    ct.weights = torch::randn({2, 4, 4});

    auto ew = ct.effective_weights();
    EXPECT_TRUE(torch::all(ew == 0).item<bool>());
}

TEST(ConnectivityTensor, EffectiveWeightsZeroModulatoryMask) {
    // When modulatory_mask is zero, effective_weights should be all zeros.
    auto ct = make_small_ct(2, 4, 8, 8);
    ct.weights = torch::randn({2, 4, 4});
    ct.modulatory_mask = torch::zeros({2, 4, 4});

    auto ew = ct.effective_weights();
    EXPECT_TRUE(torch::all(ew == 0).item<bool>());
}

TEST(ConnectivityTensor, EffectiveWeightsPartialMask) {
    // With a partial structural mask, only masked entries survive.
    auto ct = make_small_ct(1, 4, 4, 4);
    ct.weights = torch::ones({1, 4, 4});

    // Zero out half the mask.
    ct.structural_mask = torch::zeros({1, 4, 4});
    ct.structural_mask[0][0][0] = 1.0;
    ct.structural_mask[0][1][1] = 1.0;

    auto ew = ct.effective_weights();
    EXPECT_FLOAT_EQ(ew[0][0][0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(ew[0][1][1].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(ew[0][0][1].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(ew[0][2][3].item<float>(), 0.0f);
}

// ---------------------------------------------------------------------------
// total_capacity
// ---------------------------------------------------------------------------

TEST(ConnectivityTensor, TotalCapacity) {
    auto ct = make_small_ct(3, 8, 24, 24);
    EXPECT_EQ(ct.total_capacity(), 3 * 8 * 8);
}

TEST(ConnectivityTensor, TotalCapacityLargeBlocks) {
    auto ct = make_small_ct(2, 256, 512, 512);
    EXPECT_EQ(ct.total_capacity(), 2 * 256 * 256);
}

// ---------------------------------------------------------------------------
// active_count
// ---------------------------------------------------------------------------

TEST(ConnectivityTensor, ActiveCountAllActive) {
    // All structural_mask = 1 => all synapses active.
    auto ct = make_small_ct(2, 4, 8, 8, /*structural_fill=*/1.0f);
    EXPECT_EQ(ct.active_count(), 2 * 4 * 4);
}

TEST(ConnectivityTensor, ActiveCountNoneActive) {
    // All structural_mask = 0 => no synapses active.
    auto ct = make_small_ct(2, 4, 8, 8, /*structural_fill=*/0.0f);
    EXPECT_EQ(ct.active_count(), 0);
}

TEST(ConnectivityTensor, ActiveCountPartial) {
    auto ct = make_small_ct(1, 4, 4, 4, /*structural_fill=*/0.0f);
    // Set exactly 5 entries to 1.
    ct.structural_mask[0][0][0] = 1.0;
    ct.structural_mask[0][0][1] = 1.0;
    ct.structural_mask[0][1][0] = 1.0;
    ct.structural_mask[0][2][2] = 1.0;
    ct.structural_mask[0][3][3] = 1.0;

    EXPECT_EQ(ct.active_count(), 5);
}

// ---------------------------------------------------------------------------
// density
// ---------------------------------------------------------------------------

TEST(ConnectivityTensor, DensityFullyConnected) {
    auto ct = make_small_ct(2, 4, 8, 8, /*structural_fill=*/1.0f);
    EXPECT_DOUBLE_EQ(ct.density(), 1.0);
}

TEST(ConnectivityTensor, DensityEmpty) {
    auto ct = make_small_ct(2, 4, 8, 8, /*structural_fill=*/0.0f);
    EXPECT_DOUBLE_EQ(ct.density(), 0.0);
}

TEST(ConnectivityTensor, DensityHalf) {
    auto ct = make_small_ct(1, 4, 4, 4, /*structural_fill=*/0.0f);
    // Fill exactly half the entries.
    int64_t total = 4 * 4;  // 16
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (count < total / 2) {
                ct.structural_mask[0][i][j] = 1.0;
                count++;
            }
        }
    }
    EXPECT_NEAR(ct.density(), 0.5, 1e-10);
}

// ---------------------------------------------------------------------------
// Optional synapse state
// ---------------------------------------------------------------------------

TEST(ConnectivityTensor, SynapseStateMap) {
    auto ct = make_small_ct(2, 4, 8, 8);

    // Add custom synapse state tensors.
    ct.synapse_state["stp_u"] = torch::ones({2, 4, 4});
    ct.synapse_state["stp_x"] = torch::zeros({2, 4, 4});

    EXPECT_EQ(ct.synapse_state.size(), 2u);
    EXPECT_EQ(ct.synapse_state.at("stp_u").size(0), 2);
    EXPECT_EQ(ct.synapse_state.at("stp_x").size(0), 2);
}

TEST(ConnectivityTensor, TraceTensorsOptional) {
    auto ct = make_small_ct(2, 4, 8, 8);

    // By default, trace tensors should not be set.
    EXPECT_FALSE(ct.trace_pre.has_value());
    EXPECT_FALSE(ct.trace_post.has_value());

    // Set them.
    ct.trace_pre = torch::zeros({2, 4, 4});
    ct.trace_post = torch::zeros({2, 4, 4});

    EXPECT_TRUE(ct.trace_pre.has_value());
    EXPECT_TRUE(ct.trace_post.has_value());
    EXPECT_EQ(ct.trace_pre->size(0), 2);
}
