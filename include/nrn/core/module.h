#pragma once

#include <cstdint>
#include <string>

#include <torch/torch.h>

#include <nrn/core/types.h>
#include <nrn/core/state.h>

namespace nrn {

// ---------------------------------------------------------------------------
// Operations table — like struct file_operations in the Linux kernel.
//
// Every concrete module type (lif, adex, izhikevich, synapses, etc.)
// provides a static instance of this struct with function pointers to
// its implementations.
// ---------------------------------------------------------------------------
struct nrn_module_ops {
    void (*forward)(void* self, State& state, double t, double dt);
    void (*reset)(void* self);
    const char** (*state_vars)(void* self, int* count);
    int64_t (*size)(void* self);
    void (*to_device)(void* self, torch::Device device);
};

// ---------------------------------------------------------------------------
// NrnModule — type-erased handle to any module.
//
// Population and Simulation use this to dispatch without knowing the
// concrete type. Equivalent to a void* + vtable pointer, but explicit.
// ---------------------------------------------------------------------------
struct NrnModule {
    void* impl;             // concrete struct (LIFNeuron*, AdExNeuron*, etc.)
    nrn_module_ops* ops;    // dispatch table
};

// ---------------------------------------------------------------------------
// Dispatch convenience functions
// ---------------------------------------------------------------------------

inline void nrn_forward(NrnModule* m, State& state, double t, double dt) {
    m->ops->forward(m->impl, state, t, dt);
}

inline void nrn_reset(NrnModule* m) {
    m->ops->reset(m->impl);
}

inline int64_t nrn_size(NrnModule* m) {
    return m->ops->size(m->impl);
}

inline void nrn_to_device(NrnModule* m, torch::Device device) {
    m->ops->to_device(m->impl, device);
}

inline const char** nrn_state_vars(NrnModule* m, int* count) {
    return m->ops->state_vars(m->impl, count);
}

} // namespace nrn
