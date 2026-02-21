#include <gtest/gtest.h>

#include <nrn/core/types.h>

using namespace nrn;
using namespace nrn::literals;

// ---------------------------------------------------------------------------
// Unit literal conversions
// ---------------------------------------------------------------------------

TEST(Types, MillisecondLiteral) {
    Duration d = 20.0_ms;
    EXPECT_DOUBLE_EQ(d, 0.020);
}

TEST(Types, SecondLiteral) {
    Duration d = 1.0_s;
    EXPECT_DOUBLE_EQ(d, 1.0);
}

TEST(Types, MicrosecondLiteral) {
    Duration d = 100.0_us;
    EXPECT_DOUBLE_EQ(d, 100e-6);
}

TEST(Types, MillivoltLiteral) {
    double v = -65.0_mV;
    EXPECT_DOUBLE_EQ(v, -0.065);
}

TEST(Types, VoltLiteral) {
    double v = 1.0_V;
    EXPECT_DOUBLE_EQ(v, 1.0);
}

TEST(Types, NanoampereLiteral) {
    double i = 0.0805_nA;
    EXPECT_DOUBLE_EQ(i, 0.0805e-9);
}

TEST(Types, PicofaradLiteral) {
    double c = 250.0_pF;
    EXPECT_DOUBLE_EQ(c, 250e-12);
}

TEST(Types, NanosiemensLiteral) {
    double g = 4.0_nS;
    EXPECT_DOUBLE_EQ(g, 4e-9);
}

TEST(Types, MicrometreLiteral) {
    double d = 200.0_um;
    EXPECT_DOUBLE_EQ(d, 200e-6);
}

TEST(Types, NanomolarLiteral) {
    double c = 1.0_nM;
    EXPECT_DOUBLE_EQ(c, 1e-9);
}

TEST(Types, MicromolarLiteral) {
    double c = 0.5_uM;
    EXPECT_DOUBLE_EQ(c, 0.5e-6);
}

// ---------------------------------------------------------------------------
// Time and Duration arithmetic
// ---------------------------------------------------------------------------
// Since Time and Duration are both `double`, arithmetic is native.
// These tests document the intended semantics.

TEST(Types, TimePlusDuration) {
    Time t = 1.0;
    Duration dt = 0.1_ms;
    Time t2 = t + dt;
    EXPECT_DOUBLE_EQ(t2, 1.0001);
}

TEST(Types, TimeMinusTime) {
    Time t1 = 1.5;
    Time t2 = 1.0;
    Duration d = t1 - t2;
    EXPECT_DOUBLE_EQ(d, 0.5);
}

TEST(Types, DurationAddition) {
    Duration a = 10.0_ms;
    Duration b = 5.0_ms;
    EXPECT_DOUBLE_EQ(a + b, 15.0_ms);
}

TEST(Types, DurationSubtraction) {
    Duration a = 20.0_ms;
    Duration b = 5.0_ms;
    EXPECT_DOUBLE_EQ(a - b, 15.0_ms);
}

TEST(Types, DurationScaling) {
    Duration d = 10.0_ms;
    EXPECT_DOUBLE_EQ(d * 3.0, 30.0_ms);
}
