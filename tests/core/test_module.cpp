#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>

using namespace nrn;

// ---------------------------------------------------------------------------
// A trivial concrete Module for testing the CRTP base.
// ---------------------------------------------------------------------------

class TestModuleImpl : public nrn::Module<TestModuleImpl> {
public:
    explicit TestModuleImpl(int64_t n) {
        n_ = n;
        data = register_buffer("data", torch::zeros({n}));
    }

    void reset() override {
        data = torch::zeros({n_});
    }

    void forward(State& /*state*/, Time /*t*/, Duration /*dt*/) override {
        // No-op for testing.
    }

    std::vector<std::string> state_vars() const override {
        return {"data"};
    }

    torch::Tensor data;
};

TORCH_MODULE(TestModule);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(Module, Construction) {
    TestModule mod(10);
    EXPECT_EQ(mod->size(), 10);
}

TEST(Module, RegisterBufferCreatesNamedBuffer) {
    TestModule mod(5);
    // The buffer should be accessible via the module's named_buffers().
    auto buffers = mod->named_buffers();
    bool found = false;
    for (const auto& item : buffers) {
        if (item.key() == "data") {
            found = true;
            EXPECT_EQ(item.value().size(0), 5);
        }
    }
    EXPECT_TRUE(found);
}

TEST(Module, StateVars) {
    TestModule mod(8);
    auto vars = mod->state_vars();
    ASSERT_EQ(vars.size(), 1u);
    EXPECT_EQ(vars[0], "data");
}

TEST(Module, ResetZerosBuffer) {
    TestModule mod(4);
    // Modify the buffer.
    mod->data.fill_(42.0);
    EXPECT_FLOAT_EQ(mod->data[0].item<float>(), 42.0f);

    // Reset should zero it.
    mod->reset();
    EXPECT_FLOAT_EQ(mod->data[0].item<float>(), 0.0f);
}

TEST(Module, ToDevice) {
    TestModule mod(16);
    // Should start on CPU.
    EXPECT_TRUE(mod->data.device().is_cpu());

    if (torch::cuda::is_available()) {
        mod->to(torch::kCUDA);
        EXPECT_TRUE(mod->data.device().is_cuda());

        mod->to(torch::kCPU);
        EXPECT_TRUE(mod->data.device().is_cpu());
    }
}
