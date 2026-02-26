#include <gtest/gtest.h>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>
#include <nrn/core/module.h>

using namespace nrn;

// ---------------------------------------------------------------------------
// A trivial concrete module for testing the ops-table dispatch.
// ---------------------------------------------------------------------------

struct TestModuleData {
    int64_t n;
    torch::Tensor data;
};

static void test_forward(void* /*self*/, State& /*state*/, double /*t*/, double /*dt*/) {
    // No-op for testing.
}

static void test_reset(void* self) {
    auto* m = static_cast<TestModuleData*>(self);
    m->data = torch::zeros({m->n});
}

static const char* test_var_names[] = {"data"};
static const char** test_state_vars(void* /*self*/, int* count) {
    *count = 1;
    return test_var_names;
}

static int64_t test_size(void* self) {
    return static_cast<TestModuleData*>(self)->n;
}

static void test_to_device(void* self, torch::Device device) {
    auto* m = static_cast<TestModuleData*>(self);
    m->data = m->data.to(device);
}

static nrn_ops test_ops = {
    test_forward,
    test_reset,
    test_state_vars,
    test_size,
    test_to_device,
};

static TestModuleData* test_module_create(int64_t n) {
    auto* m = new TestModuleData();
    m->n = n;
    m->data = torch::zeros({n});
    return m;
}

static void test_module_destroy(TestModuleData* m) {
    delete m;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(Module, Construction) {
    auto* mod = test_module_create(10);
    auto nmod = NrnModule{mod, &test_ops};
    EXPECT_EQ(nrn_size(&nmod), 10);
    test_module_destroy(mod);
}

TEST(Module, DataTensorHasCorrectShape) {
    auto* mod = test_module_create(5);
    EXPECT_EQ(mod->data.size(0), 5);
    test_module_destroy(mod);
}

TEST(Module, StateVars) {
    auto* mod = test_module_create(8);
    auto nmod = NrnModule{mod, &test_ops};
    int count = 0;
    auto* vars = nrn_state_vars(&nmod, &count);
    ASSERT_EQ(count, 1);
    EXPECT_STREQ(vars[0], "data");
    test_module_destroy(mod);
}

TEST(Module, ResetZerosBuffer) {
    auto* mod = test_module_create(4);
    auto nmod = NrnModule{mod, &test_ops};
    // Modify the data tensor.
    mod->data.fill_(42.0);
    EXPECT_FLOAT_EQ(mod->data[0].item<float>(), 42.0f);

    // Reset should zero it.
    nrn_reset(&nmod);
    EXPECT_FLOAT_EQ(mod->data[0].item<float>(), 0.0f);
    test_module_destroy(mod);
}

TEST(Module, ToDevice) {
    auto* mod = test_module_create(16);
    auto nmod = NrnModule{mod, &test_ops};
    // Should start on CPU.
    EXPECT_TRUE(mod->data.device().is_cpu());

    if (torch::cuda::is_available()) {
        nrn_to_device(&nmod, torch::kCUDA);
        EXPECT_TRUE(mod->data.device().is_cuda());

        nrn_to_device(&nmod, torch::kCPU);
        EXPECT_TRUE(mod->data.device().is_cpu());
    }
    test_module_destroy(mod);
}
