#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/state.h>

using namespace nrn;

// ---------------------------------------------------------------------------
// state_create factory
// ---------------------------------------------------------------------------

TEST(State, CreateWithVariableNames) {
    auto state = state_create({"v", "spike"}, 100);
    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(state_contains(state, "v"));
    EXPECT_TRUE(state_contains(state, "spike"));
    EXPECT_EQ(state_get(state, "v").size(0), 100);
    EXPECT_EQ(state_get(state, "spike").size(0), 100);
}

TEST(State, CreateTensorsAreZero) {
    auto state = state_create({"v"}, 10);
    auto v = state_get(state, "v");
    EXPECT_TRUE(torch::all(v == 0).item<bool>());
}

// ---------------------------------------------------------------------------
// state_set / state_get / operator[]
// ---------------------------------------------------------------------------

TEST(State, SetAndGet) {
    State state;
    auto t = torch::ones({5});
    state_set(state, "x", t);
    EXPECT_TRUE(torch::equal(state_get(state, "x"), t));
}

TEST(State, BracketOperator) {
    State state;
    state_set(state, "v", torch::zeros({10}));
    auto& v = state["v"];
    EXPECT_EQ(v.size(0), 10);
}

TEST(State, GetThrowsOnMissing) {
    State state;
    EXPECT_THROW(state_get(state, "nonexistent"), c10::Error);
}

// ---------------------------------------------------------------------------
// state_contains
// ---------------------------------------------------------------------------

TEST(State, ContainsReturnsTrue) {
    State state;
    state_set(state, "v", torch::zeros({1}));
    EXPECT_TRUE(state_contains(state, "v"));
}

TEST(State, ContainsReturnsFalse) {
    State state;
    EXPECT_FALSE(state_contains(state, "nonexistent"));
}

// ---------------------------------------------------------------------------
// state_remove
// ---------------------------------------------------------------------------

TEST(State, RemoveExistingTensor) {
    State state;
    state_set(state, "v", torch::zeros({5}));
    EXPECT_EQ(state.size(), 1u);
    bool removed = state_remove(state, "v");
    EXPECT_TRUE(removed);
    EXPECT_EQ(state.size(), 0u);
    EXPECT_FALSE(state_contains(state, "v"));
}

TEST(State, RemoveNonexistentReturnsFalse) {
    State state;
    EXPECT_FALSE(state_remove(state, "nope"));
}

// ---------------------------------------------------------------------------
// size / state_keys
// ---------------------------------------------------------------------------

TEST(State, SizeIsCorrect) {
    State state;
    EXPECT_EQ(state.size(), 0u);
    state_set(state, "a", torch::zeros({1}));
    state_set(state, "b", torch::zeros({1}));
    EXPECT_EQ(state.size(), 2u);
}

TEST(State, KeysContainAllNames) {
    State state;
    state_set(state, "alpha", torch::zeros({1}));
    state_set(state, "beta", torch::zeros({1}));
    auto keys = state_keys(state);
    EXPECT_EQ(keys.size(), 2u);
    // Order is unspecified (unordered_map), so sort before comparing.
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys[0], "alpha");
    EXPECT_EQ(keys[1], "beta");
}

// ---------------------------------------------------------------------------
// state_to_device — move tensors between devices
// ---------------------------------------------------------------------------

TEST(State, ToDevice) {
    auto state = state_create({"v", "spike"}, 50);
    // Tensors should start on CPU.
    EXPECT_TRUE(state_get(state, "v").device().is_cpu());

    // If CUDA is available, move to GPU and verify.
    if (torch::cuda::is_available()) {
        state_to_device(state, torch::kCUDA);
        EXPECT_TRUE(state_get(state, "v").device().is_cuda());
        EXPECT_TRUE(state_get(state, "spike").device().is_cuda());

        // Move back to CPU.
        state_to_device(state, torch::kCPU);
        EXPECT_TRUE(state_get(state, "v").device().is_cpu());
    }
}
