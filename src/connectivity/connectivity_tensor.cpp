#include <nrn/connectivity/connectivity_tensor.h>

namespace nrn {

// ---------------------------------------------------------------------------
// BlockIndex
// ---------------------------------------------------------------------------

int64_t BlockIndex::find_block(int64_t target_row, int64_t source_col) const {
    TORCH_CHECK(row_ptr.defined() && col_idx.defined(),
                "BlockIndex: row_ptr and col_idx must be defined");
    TORCH_CHECK(target_row >= 0 && target_row < n_rows(),
                "BlockIndex::find_block: target_row out of range");

    auto rp = row_ptr.accessor<int32_t, 1>();
    auto ci = col_idx.accessor<int32_t, 1>();

    int32_t start = rp[target_row];
    int32_t end   = rp[target_row + 1];

    for (int32_t i = start; i < end; ++i) {
        if (ci[i] == static_cast<int32_t>(source_col)) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// ConnectivityTensor
// ---------------------------------------------------------------------------

torch::Tensor ConnectivityTensor::effective_weights() const {
    TORCH_CHECK(weights.defined(), "ConnectivityTensor: weights not defined");
    TORCH_CHECK(structural_mask.defined(),
                "ConnectivityTensor: structural_mask not defined");
    TORCH_CHECK(modulatory_mask.defined(),
                "ConnectivityTensor: modulatory_mask not defined");

    // Cast masks to the weight dtype so the element-wise products work
    // even when masks are stored in a narrower type (e.g., float16).
    auto s_mask = structural_mask.to(weights.dtype());
    auto m_mask = modulatory_mask.to(weights.dtype());

    return weights * s_mask * m_mask;
}

void ConnectivityTensor::to(torch::Device device) {
    if (block_index.row_ptr.defined())
        block_index.row_ptr = block_index.row_ptr.to(device);
    if (block_index.col_idx.defined())
        block_index.col_idx = block_index.col_idx.to(device);
    if (weights.defined())
        weights = weights.to(device);
    if (structural_mask.defined())
        structural_mask = structural_mask.to(device);
    if (modulatory_mask.defined())
        modulatory_mask = modulatory_mask.to(device);
    if (delays.defined())
        delays = delays.to(device);

    for (auto& [name, tensor] : synapse_state) {
        if (tensor.defined())
            tensor = tensor.to(device);
    }
    if (trace_pre.has_value() && trace_pre->defined())
        trace_pre = trace_pre->to(device);
    if (trace_post.has_value() && trace_post->defined())
        trace_post = trace_post->to(device);
}

int64_t ConnectivityTensor::total_capacity() const {
    return block_index.n_blocks() * block_size * block_size;
}

int64_t ConnectivityTensor::active_count() const {
    if (!structural_mask.defined()) {
        return total_capacity();
    }
    // Count non-zero entries in the structural mask.
    return structural_mask.to(torch::kFloat32).sum().item<int64_t>();
}

double ConnectivityTensor::density() const {
    int64_t cap = total_capacity();
    if (cap == 0) return 0.0;
    return static_cast<double>(active_count()) / static_cast<double>(cap);
}

} // namespace nrn
