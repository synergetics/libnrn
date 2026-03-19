#include <cuda_runtime.h>
#include <cstdint>

namespace nrn {
namespace cuda {

// ---------------------------------------------------------------------------
// Block-dense masked matmul kernel for synaptic current delivery.
//
// One CUDA thread block per dense block in the CSR structure.
// Each thread computes one row of the block:
//     I_syn[target_neuron] += sum_k W_eff[row][k] * spike[source_neuron_k]
//
// Uses shared memory to cache the source spike sub-vector per block.
// ---------------------------------------------------------------------------

// Maximum block size we support in shared memory (1024 floats = 4 KB).
static constexpr int MAX_BLOCK_SIZE = 1024;

__global__ void synaptic_current_block_dense_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ structural_mask,
    const float* __restrict__ modulatory_mask,
    const float* __restrict__ spikes,
    float*       __restrict__ I_syn,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    int32_t block_size,
    int32_t n_block_rows,
    int32_t n_source,
    int32_t n_target) {

    // Each block in the grid handles one non-zero block in the CSR structure.
    // We flatten the iteration: blockIdx.x indexes into the total nnz blocks.
    // We need to find which target block row this belongs to.

    int32_t block_idx = blockIdx.x;

    // Find target block row via binary search on row_ptr.
    int32_t lo = 0, hi = n_block_rows;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (row_ptr[mid + 1] <= block_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int32_t target_block_row = lo;
    int32_t source_block_col = col_idx[block_idx];

    int32_t t_begin = target_block_row * block_size;
    int32_t s_begin = source_block_col * block_size;

    // Clamp to actual dimensions.
    int32_t t_size = min(block_size, n_target - t_begin);
    int32_t s_size = min(block_size, n_source - s_begin);

    if (t_size <= 0 || s_size <= 0) return;

    // Load source spike sub-vector into shared memory.
    extern __shared__ float shared_spikes[];

    for (int32_t i = threadIdx.x; i < s_size; i += blockDim.x) {
        shared_spikes[i] = spikes[s_begin + i];
    }
    __syncthreads();

    // Each thread computes one row of the masked matmul.
    int32_t row = threadIdx.x;
    if (row >= t_size) return;

    // Pointer to this block's data: block_idx * B * B + row * B
    int64_t block_offset = (int64_t)block_idx * block_size * block_size
                         + (int64_t)row * block_size;

    float acc = 0.0f;
    for (int32_t k = 0; k < s_size; ++k) {
        float w = weights[block_offset + k];
        float ms = structural_mask[block_offset + k];
        float mm = modulatory_mask[block_offset + k];
        acc += w * ms * mm * shared_spikes[k];
    }

    // Atomically accumulate into I_syn (multiple blocks may target same neurons).
    atomicAdd(&I_syn[t_begin + row], acc);
}

// ---------------------------------------------------------------------------
// Host-side dispatch.
// ---------------------------------------------------------------------------
void launch_synaptic_current_block_dense(
    const float* weights,
    const float* structural_mask,
    const float* modulatory_mask,
    const float* spikes,
    float*       I_syn,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    int32_t block_size,
    int32_t n_block_rows,
    int32_t n_blocks,
    int32_t n_source,
    int32_t n_target,
    void* stream_ptr) {

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (n_blocks == 0) return;

    // One CUDA thread block per CSR non-zero block.
    // Threads per block = block_size (one thread per target row in the dense block).
    int threads = min(block_size, 1024);
    int shared_mem = block_size * sizeof(float);

    synaptic_current_block_dense_kernel<<<n_blocks, threads, shared_mem, stream>>>(
        weights, structural_mask, modulatory_mask,
        spikes, I_syn,
        row_ptr, col_idx,
        block_size, n_block_rows,
        n_source, n_target);
}

} // namespace cuda
} // namespace nrn
