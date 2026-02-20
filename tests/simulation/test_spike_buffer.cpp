#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/simulation/spike_buffer.h>

using namespace nrn;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, Construction) {
    SpikeBuffer buf(100, 10, torch::kCPU);
    EXPECT_EQ(buf.size(), 100);
    EXPECT_EQ(buf.max_delay(), 10);
}

TEST(SpikeBuffer, ConstructionSmall) {
    SpikeBuffer buf(1, 1, torch::kCPU);
    EXPECT_EQ(buf.size(), 1);
    EXPECT_EQ(buf.max_delay(), 1);
}

TEST(SpikeBuffer, ConstructionLarge) {
    SpikeBuffer buf(10000, 50, torch::kCPU);
    EXPECT_EQ(buf.size(), 10000);
    EXPECT_EQ(buf.max_delay(), 50);
}

// ---------------------------------------------------------------------------
// Push and read with delay=1 (most recent push)
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, PushAndReadDelay1) {
    SpikeBuffer buf(5, 4, torch::kCPU);

    // Push a spike vector: neurons 0 and 3 fire.
    auto spikes = torch::zeros({5});
    spikes[0] = 1.0;
    spikes[3] = 1.0;
    buf.push(spikes);

    // Read with delay=1 should return the most recently pushed vector.
    auto out = buf.read(1);
    EXPECT_EQ(out.size(0), 5);
    EXPECT_FLOAT_EQ(out[0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(out[1].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(out[3].item<float>(), 1.0f);
}

TEST(SpikeBuffer, PushAndReadAllZeros) {
    SpikeBuffer buf(4, 3, torch::kCPU);

    auto spikes = torch::zeros({4});
    buf.push(spikes);

    auto out = buf.read(1);
    EXPECT_TRUE(torch::all(out == 0).item<bool>());
}

TEST(SpikeBuffer, PushAndReadAllOnes) {
    SpikeBuffer buf(4, 3, torch::kCPU);

    auto spikes = torch::ones({4});
    buf.push(spikes);

    auto out = buf.read(1);
    EXPECT_TRUE(torch::all(out == 1).item<bool>());
}

// ---------------------------------------------------------------------------
// Push multiple and read with different delays
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, MultipleDelays) {
    SpikeBuffer buf(3, 4, torch::kCPU);

    // Step 1: neuron 0 fires.
    auto s1 = torch::zeros({3});
    s1[0] = 1.0;
    buf.push(s1);

    // Step 2: neuron 1 fires.
    auto s2 = torch::zeros({3});
    s2[1] = 1.0;
    buf.push(s2);

    // delay=1 should give the most recent push (step 2: neuron 1).
    auto d1 = buf.read(1);
    EXPECT_FLOAT_EQ(d1[1].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(d1[0].item<float>(), 0.0f);

    // delay=2 should give the earlier push (step 1: neuron 0).
    auto d2 = buf.read(2);
    EXPECT_FLOAT_EQ(d2[0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(d2[1].item<float>(), 0.0f);
}

TEST(SpikeBuffer, ThreeStepsThreeDelays) {
    SpikeBuffer buf(3, 4, torch::kCPU);

    // Step 1: neuron 0.
    auto s1 = torch::zeros({3});
    s1[0] = 1.0;
    buf.push(s1);

    // Step 2: neuron 1.
    auto s2 = torch::zeros({3});
    s2[1] = 1.0;
    buf.push(s2);

    // Step 3: neuron 2.
    auto s3 = torch::zeros({3});
    s3[2] = 1.0;
    buf.push(s3);

    // delay=1 => step 3 (neuron 2).
    auto d1 = buf.read(1);
    EXPECT_FLOAT_EQ(d1[2].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(d1[0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(d1[1].item<float>(), 0.0f);

    // delay=2 => step 2 (neuron 1).
    auto d2 = buf.read(2);
    EXPECT_FLOAT_EQ(d2[1].item<float>(), 1.0f);

    // delay=3 => step 1 (neuron 0).
    auto d3 = buf.read(3);
    EXPECT_FLOAT_EQ(d3[0].item<float>(), 1.0f);
}

// ---------------------------------------------------------------------------
// Wrap-around: push more than max_delay times
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, WrapAround) {
    int64_t max_delay = 4;
    SpikeBuffer buf(2, max_delay, torch::kCPU);

    // Push max_delay + 2 vectors to force wrap-around.
    for (int i = 0; i < max_delay + 2; ++i) {
        auto s = torch::zeros({2});
        s[0] = static_cast<float>(i % 2);  // alternating
        buf.push(s);
    }

    // After 6 pushes (indices 0..5) with max_delay=4:
    // The buffer should contain the last 4 pushes (indices 2,3,4,5).
    // delay=1 is the most recent (index 5): s[0] = 5%2 = 1.
    auto latest = buf.read(1);
    EXPECT_FLOAT_EQ(latest[0].item<float>(), 1.0f);

    // delay=2 is index 4: s[0] = 4%2 = 0.
    auto prev = buf.read(2);
    EXPECT_FLOAT_EQ(prev[0].item<float>(), 0.0f);
}

TEST(SpikeBuffer, WrapAroundManyTimes) {
    int64_t max_delay = 3;
    SpikeBuffer buf(1, max_delay, torch::kCPU);

    // Push 20 vectors, each with value = push_count (as float).
    for (int i = 0; i < 20; ++i) {
        auto s = torch::full({1}, static_cast<float>(i));
        buf.push(s);
    }

    // delay=1 should be the most recent push (index 19).
    auto d1 = buf.read(1);
    EXPECT_FLOAT_EQ(d1[0].item<float>(), 19.0f);

    // delay=2 should be index 18.
    auto d2 = buf.read(2);
    EXPECT_FLOAT_EQ(d2[0].item<float>(), 18.0f);

    // delay=3 should be index 17.
    auto d3 = buf.read(3);
    EXPECT_FLOAT_EQ(d3[0].item<float>(), 17.0f);
}

// ---------------------------------------------------------------------------
// read_batch: per-neuron delays
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, ReadBatchBasic) {
    SpikeBuffer buf(3, 4, torch::kCPU);

    // Push 3 steps.
    for (int step = 0; step < 3; ++step) {
        auto s = torch::zeros({3});
        s[step] = 1.0;  // neuron `step` fires at step `step`.
        buf.push(s);
    }

    // Per-neuron delays: neuron 0 looks back 3 steps, neuron 1 looks back 2,
    // neuron 2 looks back 1.
    auto delays = torch::tensor({3L, 2L, 1L});
    auto out = buf.read_batch(delays);

    EXPECT_EQ(out.size(0), 3);
    // Neuron 0 at delay=3 should see step 0 => neuron 0 fired => 1.0.
    EXPECT_FLOAT_EQ(out[0].item<float>(), 1.0f);
    // Neuron 1 at delay=2 should see step 1 => neuron 1 fired => 1.0.
    EXPECT_FLOAT_EQ(out[1].item<float>(), 1.0f);
    // Neuron 2 at delay=1 should see step 2 => neuron 2 fired => 1.0.
    EXPECT_FLOAT_EQ(out[2].item<float>(), 1.0f);
}

TEST(SpikeBuffer, ReadBatchUniformDelay) {
    SpikeBuffer buf(4, 5, torch::kCPU);

    auto s = torch::ones({4});
    buf.push(s);

    // All neurons look back 1 step.
    auto delays = torch::ones({4}, torch::kInt64);
    auto out = buf.read_batch(delays);

    EXPECT_TRUE(torch::all(out == 1.0).item<bool>());
}

// ---------------------------------------------------------------------------
// Reset clears buffer
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, ResetClearsBuffer) {
    SpikeBuffer buf(3, 4, torch::kCPU);

    auto s = torch::ones({3});
    buf.push(s);

    buf.reset();

    // After reset, reading any delay should return zeros.
    auto out = buf.read(1);
    EXPECT_TRUE(torch::all(out == 0).item<bool>());
}

TEST(SpikeBuffer, ResetAllowsReuse) {
    SpikeBuffer buf(2, 3, torch::kCPU);

    // Push some data.
    buf.push(torch::ones({2}));
    buf.push(torch::zeros({2}));

    // Reset.
    buf.reset();

    // Push new data and verify it works correctly.
    auto s = torch::zeros({2});
    s[1] = 1.0;
    buf.push(s);

    auto out = buf.read(1);
    EXPECT_FLOAT_EQ(out[0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(out[1].item<float>(), 1.0f);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(SpikeBuffer, ReadBeforePush) {
    SpikeBuffer buf(4, 5, torch::kCPU);

    // Reading before any push — buffer is all zeros.
    auto out = buf.read(1);
    EXPECT_TRUE(torch::all(out == 0).item<bool>());
}

TEST(SpikeBuffer, MaxDelayRead) {
    SpikeBuffer buf(2, 3, torch::kCPU);

    // Push exactly max_delay vectors.
    for (int i = 0; i < 3; ++i) {
        auto s = torch::full({2}, static_cast<float>(i + 1));
        buf.push(s);
    }

    // Read at max delay.
    auto out = buf.read(3);
    EXPECT_FLOAT_EQ(out[0].item<float>(), 1.0f);  // First push.
}
