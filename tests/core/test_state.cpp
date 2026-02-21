#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/state.h>

using namespace nrn;

// ---------------------------------------------------------------------------
// State::create factory
// ---------------------------------------------------------------------------

TEST(State, CreateWithVariableNames) {
    auto state = State::create({"v", "spike"}, 100);
    EXPECT_EQ(state.size(), 2u);
    EXPECT_TRUE(state.has("v"));
    EXPECT_TRUE(state.has("spike"));
    EXPECT_EQ(state.get("v").size(0), 100);
    EXPECT_EQ(state.get("spike").size(0), 100);
}

TEST(State, CreateTensorsAreZero) {
    auto state = State::create({"v"}, 10);
    auto v = state.get("v");
    EXPECT_TRUE(torch::all(v == 0).item<bool>());
}

// ---------------------------------------------------------------------------
// set / get / operator[]
// ---------------------------------------------------------------------------

TEST(State, SetAndGet) {
    State state;
    auto t = torch::ones({5});
    state.set("x", t);
    EXPECT_TRUE(torch::equal(state.get("x"), t));
}

TEST(State, BracketOperator) {
    State state;
    state.set("v", torch::zeros({10}));
    auto& v = state["v"];
    EXPECT_EQ(v.size(0), 10);
}

TEST(State, GetThrowsOnMissing) {
    State state;
    EXPECT_THROW(state.get("nonexistent"), c10::Error);
}

// ---------------------------------------------------------------------------
// has / contains
// ---------------------------------------------------------------------------

TEST(State, HasReturnsTrue) {
    State state;
    state.set("v", torch::zeros({1}));
    EXPECT_TRUE(state.has("v"));
    EXPECT_TRUE(state.contains("v"));
}

TEST(State, HasReturnsFalse) {
    State state;
    EXPECT_FALSE(state.has("nonexistent"));
    EXPECT_FALSE(state.contains("nonexistent"));
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

TEST(State, RemoveExistingTensor) {
    State state;
    state.set("v", torch::zeros({5}));
    EXPECT_EQ(state.size(), 1u);
    bool removed = state.remove("v");
    EXPECT_TRUE(removed);
    EXPECT_EQ(state.size(), 0u);
    EXPECT_FALSE(state.has("v"));
}

TEST(State, RemoveNonexistentReturnsFalse) {
    State state;
    EXPECT_FALSE(state.remove("nope"));
}

// ---------------------------------------------------------------------------
// size / keys
// ---------------------------------------------------------------------------

TEST(State, SizeIsCorrect) {
    State state;
    EXPECT_EQ(state.size(), 0u);
    state.set("a", torch::zeros({1}));
    state.set("b", torch::zeros({1}));
    EXPECT_EQ(state.size(), 2u);
}

TEST(State, KeysContainAllNames) {
    State state;
    state.set("alpha", torch::zeros({1}));
    state.set("beta", torch::zeros({1}));
    auto keys = state.keys();
    EXPECT_EQ(keys.size(), 2u);
    // Order is unspecified (unordered_map), so sort before comparing.
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys[0], "alpha");
    EXPECT_EQ(keys[1], "beta");
}

// ---------------------------------------------------------------------------
// to(device) — move tensors between devices
// ---------------------------------------------------------------------------

TEST(State, ToDevice) {
    auto state = State::create({"v", "spike"}, 50);
    // Tensors should start on CPU.
    EXPECT_TRUE(state.get("v").device().is_cpu());

    // If CUDA is available, move to GPU and verify.
    if (torch::cuda::is_available()) {
        state.to(torch::kCUDA);
        EXPECT_TRUE(state.get("v").device().is_cuda());
        EXPECT_TRUE(state.get("spike").device().is_cuda());

        // Move back to CPU.
        state.to(torch::kCPU);
        EXPECT_TRUE(state.get("v").device().is_cpu());
    }
}
