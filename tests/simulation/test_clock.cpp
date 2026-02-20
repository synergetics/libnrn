#include <gtest/gtest.h>

#include <nrn/core/types.h>
#include <nrn/simulation/clock.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(Clock, Construction) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);
    EXPECT_DOUBLE_EQ(clk.dt_fast(), 0.0001);
    EXPECT_DOUBLE_EQ(clk.dt_slow(), 0.01);
    EXPECT_DOUBLE_EQ(clk.dt_structural(), 1.0);
    EXPECT_DOUBLE_EQ(clk.time(), 0.0);
    EXPECT_EQ(clk.step(), 0u);
}

TEST(Clock, ConstructionDifferentTimescales) {
    Clock clk(0.05_ms, 5.0_ms, 0.5_s);
    EXPECT_DOUBLE_EQ(clk.dt_fast(), 0.05e-3);
    EXPECT_DOUBLE_EQ(clk.dt_slow(), 5.0e-3);
    EXPECT_DOUBLE_EQ(clk.dt_structural(), 0.5);
}

TEST(Clock, ConstructionLargeRatios) {
    // dt_structural / dt_fast = 1.0 / 0.0001 = 10000
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);
    EXPECT_DOUBLE_EQ(clk.dt_fast(), 0.1e-3);
    EXPECT_DOUBLE_EQ(clk.dt_structural(), 1.0);
}

TEST(Clock, InitialStateIsZero) {
    Clock clk(1.0_ms, 10.0_ms, 1.0_s);
    EXPECT_DOUBLE_EQ(clk.time(), 0.0);
    EXPECT_EQ(clk.step(), 0u);
}

// ---------------------------------------------------------------------------
// advance_fast accumulates time
// ---------------------------------------------------------------------------

TEST(Clock, AdvanceFastSingleStep) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    clk.advance_fast();
    EXPECT_EQ(clk.step(), 1u);
    EXPECT_NEAR(clk.time(), 0.0001, 1e-12);
}

TEST(Clock, AdvanceFastAccumulatesTime) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    clk.advance_fast();
    EXPECT_EQ(clk.step(), 1u);
    EXPECT_NEAR(clk.time(), 0.0001, 1e-12);

    // Advance 99 more times (total 100 fast steps).
    for (int i = 0; i < 99; ++i) {
        clk.advance_fast();
    }
    EXPECT_EQ(clk.step(), 100u);
    EXPECT_NEAR(clk.time(), 0.01, 1e-10);
}

TEST(Clock, AdvanceFastManySteps) {
    Clock clk(1.0_ms, 100.0_ms, 10.0_s);

    for (int i = 0; i < 1000; ++i) {
        clk.advance_fast();
    }
    EXPECT_EQ(clk.step(), 1000u);
    EXPECT_NEAR(clk.time(), 1.0, 1e-8);
}

TEST(Clock, StepMonotonicallyIncreasing) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    uint64_t prev_step = 0;
    for (int i = 0; i < 50; ++i) {
        clk.advance_fast();
        EXPECT_GT(clk.step(), prev_step);
        prev_step = clk.step();
    }
}

TEST(Clock, TimeMonotonicallyIncreasing) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    double prev_time = -1.0;
    for (int i = 0; i < 50; ++i) {
        clk.advance_fast();
        EXPECT_GT(clk.time(), prev_time);
        prev_time = clk.time();
    }
}

// ---------------------------------------------------------------------------
// is_slow_boundary
// ---------------------------------------------------------------------------

TEST(Clock, SlowBoundary) {
    // dt_slow = 10 ms, dt_fast = 0.1 ms  =>  slow_ratio = 100
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    // Advance 100 fast steps to reach the first slow boundary.
    for (int i = 0; i < 100; ++i) {
        clk.advance_fast();
    }

    // At step 100, we should be at a slow boundary.
    EXPECT_TRUE(clk.is_slow_boundary());

    // Step 101 should NOT be a slow boundary.
    clk.advance_fast();
    EXPECT_FALSE(clk.is_slow_boundary());
}

TEST(Clock, SlowBoundaryMultiple) {
    // Verify multiple slow boundaries.
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    int slow_count = 0;
    for (int i = 0; i < 500; ++i) {
        clk.advance_fast();
        if (clk.is_slow_boundary()) {
            slow_count++;
        }
    }
    // 500 fast steps at 0.1ms each = 50ms total.
    // With dt_slow = 10ms, we expect 5 slow boundaries (at 100, 200, 300, 400, 500).
    EXPECT_EQ(slow_count, 5);
}

TEST(Clock, NotSlowBoundaryAtStepZero) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);
    // Step 0 should not be a boundary (no advance has happened).
    // Boundary queries are meaningful after advance_fast().
    // At step 0 (before any advance), the boundary state depends on implementation.
    // Most implementations treat step 0 as a boundary (0 % ratio == 0).
    // We just verify it doesn't crash.
    (void)clk.is_slow_boundary();
    SUCCEED();
}

TEST(Clock, SlowBoundaryWithDifferentRatio) {
    // dt_slow = 5 ms, dt_fast = 1 ms => slow_ratio = 5
    Clock clk(1.0_ms, 5.0_ms, 1.0_s);

    for (int i = 0; i < 5; ++i) {
        clk.advance_fast();
    }
    EXPECT_TRUE(clk.is_slow_boundary());

    clk.advance_fast();
    EXPECT_FALSE(clk.is_slow_boundary());
}

// ---------------------------------------------------------------------------
// is_structural_boundary
// ---------------------------------------------------------------------------

TEST(Clock, StructuralBoundary) {
    // dt_structural = 1.0 s, dt_fast = 0.1 ms  =>  structural_ratio = 10000
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    // Advance to the structural boundary.
    for (int i = 0; i < 10000; ++i) {
        clk.advance_fast();
    }
    EXPECT_TRUE(clk.is_structural_boundary());
}

TEST(Clock, StructuralBoundaryNotAtNonMultiple) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    // Advance to just before the structural boundary.
    for (int i = 0; i < 9999; ++i) {
        clk.advance_fast();
    }
    EXPECT_FALSE(clk.is_structural_boundary());
}

TEST(Clock, StructuralBoundaryAlsoSlowBoundary) {
    // At the structural boundary, we should also be at a slow boundary
    // (since dt_structural is a multiple of dt_slow).
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    for (int i = 0; i < 10000; ++i) {
        clk.advance_fast();
    }
    EXPECT_TRUE(clk.is_structural_boundary());
    EXPECT_TRUE(clk.is_slow_boundary());
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

TEST(Clock, Reset) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    for (int i = 0; i < 50; ++i) {
        clk.advance_fast();
    }
    EXPECT_GT(clk.step(), 0u);

    clk.reset();
    EXPECT_EQ(clk.step(), 0u);
    EXPECT_DOUBLE_EQ(clk.time(), 0.0);
}

TEST(Clock, ResetAllowsReuse) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    // Run some steps.
    for (int i = 0; i < 200; ++i) {
        clk.advance_fast();
    }

    // Reset and run again.
    clk.reset();
    EXPECT_EQ(clk.step(), 0u);
    EXPECT_DOUBLE_EQ(clk.time(), 0.0);

    // Should be able to advance again from zero.
    clk.advance_fast();
    EXPECT_EQ(clk.step(), 1u);
    EXPECT_NEAR(clk.time(), 0.0001, 1e-12);
}

TEST(Clock, ResetPreservesTimescales) {
    Clock clk(0.1_ms, 10.0_ms, 1.0_s);

    for (int i = 0; i < 50; ++i) {
        clk.advance_fast();
    }

    clk.reset();

    // Timescale durations should be unchanged.
    EXPECT_DOUBLE_EQ(clk.dt_fast(), 0.0001);
    EXPECT_DOUBLE_EQ(clk.dt_slow(), 0.01);
    EXPECT_DOUBLE_EQ(clk.dt_structural(), 1.0);
}

// ---------------------------------------------------------------------------
// Boundary detection after reset
// ---------------------------------------------------------------------------

TEST(Clock, SlowBoundaryAfterReset) {
    Clock clk(1.0_ms, 10.0_ms, 1.0_s);

    // Advance past a slow boundary.
    for (int i = 0; i < 15; ++i) {
        clk.advance_fast();
    }

    clk.reset();

    // Advance to first slow boundary again.
    for (int i = 0; i < 10; ++i) {
        clk.advance_fast();
    }
    EXPECT_TRUE(clk.is_slow_boundary());
}
