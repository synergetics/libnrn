#include <nrn/plasticity/stdp.h>
#include <nrn/connectivity/connectivity_tensor.h>

#include <algorithm>
#include <cmath>

namespace nrn {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

STDPState* stdp_create(const STDPOptions& opts) {
    auto* s = new STDPState{};
    s->opts = opts;
    s->cached_dt  = 0.0;
    s->decay_pre  = 0.0;
    s->decay_post = 0.0;
    return s;
}

void stdp_destroy(STDPState* s) {
    delete s;
}

// ---------------------------------------------------------------------------
// Ops implementations (void* self -> STDPState*)
// ---------------------------------------------------------------------------

void stdp_initialize(void* self, ConnectivityTensor& conn) {
    // self is unused for initialization state, but we validate weights.
    (void)self;

    TORCH_CHECK(conn.weights.defined(),
                "stdp_initialize: weights tensor must be defined");

    // Allocate trace tensors with the same shape and device as weights,
    // initialized to zero.
    conn.trace_pre  = torch::zeros_like(conn.weights);
    conn.trace_post = torch::zeros_like(conn.weights);
}

void stdp_update(void* self, ConnectivityTensor& conn,
                 const State& pre_state, const State& post_state,
                 double /*t*/, double dt) {
    auto* s = static_cast<STDPState*>(self);

    // Cache decay factors when dt changes.
    if (dt != s->cached_dt) {
        s->cached_dt  = dt;
        s->decay_pre  = std::exp(-dt / s->opts.tau_plus());
        s->decay_post = std::exp(-dt / s->opts.tau_minus());
    }

    // Need spike tensors from pre and post populations.
    if (!state_contains(pre_state, "spike") ||
        !state_contains(post_state, "spike")) {
        return;
    }

    // Traces must have been initialized by stdp_initialize().
    if (!conn.trace_pre.has_value() || !conn.trace_post.has_value()) {
        return;
    }

    auto pre_spikes  = state_get(pre_state, "spike");   // [N_pre]
    auto post_spikes = state_get(post_state, "spike");   // [N_post]

    auto& trace_pre  = conn.trace_pre.value();   // [n_blocks, B, B]
    auto& trace_post = conn.trace_post.value();  // [n_blocks, B, B]

    const auto& bi = conn.block_index;
    int64_t B = conn.block_size;
    int64_t n_src = conn.n_source;
    int64_t n_tgt = conn.n_target;
    int64_t n_tgt_blocks = bi.n_rows();

    auto row_ptr_cpu = bi.row_ptr.to(torch::kCPU);
    auto col_idx_cpu = bi.col_idx.to(torch::kCPU);
    auto rp = row_ptr_cpu.accessor<int32_t, 1>();
    auto ci = col_idx_cpu.accessor<int32_t, 1>();

    double a_plus  = s->opts.a_plus();
    double a_minus = std::abs(s->opts.a_minus());
    double lr      = s->opts.learning_rate();
    double w_min   = s->opts.w_min();
    double w_max   = s->opts.w_max();

    for (int64_t tr = 0; tr < n_tgt_blocks; ++tr) {
        int32_t block_start = rp[tr];
        int32_t block_end   = rp[tr + 1];

        int64_t t_begin = tr * B;
        int64_t t_end   = std::min(t_begin + B, n_tgt);
        int64_t t_size  = t_end - t_begin;

        // Post spikes for this target block row — [t_size]
        auto post_blk = post_spikes.slice(0, t_begin, t_end);

        for (int32_t bi_idx = block_start; bi_idx < block_end; ++bi_idx) {
            int32_t sc = ci[bi_idx];

            int64_t s_begin = sc * B;
            int64_t s_end   = std::min(s_begin + B, n_src);
            int64_t s_size  = s_end - s_begin;

            // Pre spikes for this source block col — [s_size]
            auto pre_blk = pre_spikes.slice(0, s_begin, s_end);

            // Slice block tensors to actual size (handle edge blocks)
            auto tp = trace_pre[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);
            auto tq = trace_post[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);
            auto W  = conn.weights[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);
            auto sm = conn.structural_mask[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);
            auto mm = conn.modulatory_mask[bi_idx].slice(0, 0, t_size).slice(1, 0, s_size);

            // 1. Decay traces
            tp.mul_(s->decay_pre);
            tq.mul_(s->decay_post);

            // 2. On pre spikes: increment trace_pre rows
            //    pre_blk is [s_size], broadcast to [t_size, s_size]
            auto pre_row = pre_blk.unsqueeze(0);   // [1, s_size]
            tp.add_(pre_row);                       // broadcast adds to all rows

            // 3. On post spikes: increment trace_post columns
            auto post_col = post_blk.unsqueeze(1);  // [t_size, 1]
            tq.add_(post_col);                       // broadcast adds to all cols

            // 4. Compute dw:
            //    LTP: a_plus * trace_pre * post_spike (post_col broadcast)
            //    LTD: a_minus * trace_post * pre_spike (pre_row broadcast)
            auto dw = a_plus * tp * post_col - a_minus * tq * pre_row;

            // 5. Apply weight update, gated by masks
            W.add_(dw * lr * sm * mm);

            // 6. Clamp weights
            W.clamp_(w_min, w_max);
        }
    }
}

void stdp_reset(void* self) {
    auto* s = static_cast<STDPState*>(self);

    // Traces live inside the ConnectivityTensor, not here.
    // A full reset requires access to the ConnectivityTensor:
    //   conn.trace_pre->zero_();
    //   conn.trace_post->zero_();
    // This function resets only local cached state.
    s->cached_dt  = 0.0;
    s->decay_pre  = 0.0;
    s->decay_post = 0.0;
}

// ---------------------------------------------------------------------------
// Read-only accessor
// ---------------------------------------------------------------------------

const STDPOptions& stdp_options(const STDPState* s) {
    return s->opts;
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

plasticity_ops stdp_ops = {
    stdp_initialize,
    stdp_update,
    stdp_reset,
};

} // namespace nrn
