#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/**
 * Fused RMSNorm Native CUDA Kernel
 * 
 * Computes:
 * RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
 * Directly in registers using warp reductions for zero memory round-trips.
 */

#define WARP_SIZE 32

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_forward_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ Weight,
    __nv_bfloat16* __restrict__ Y,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const __nv_bfloat16* x_row = X + row * hidden_dim;
    __nv_bfloat16* y_row = Y + row * hidden_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        sum_sq += val * val;
    }

    sum_sq = warp_sum(sum_sq);

    __shared__ float s_warp_sums[32]; // Max 32 warps (1024 threads)
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        s_warp_sums[wid] = sum_sq;
    }
    __syncthreads();

    __shared__ float s_mean_sq;
    if (tid == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float total_sum_sq = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            total_sum_sq += s_warp_sums[w];
        }
        s_mean_sq = rsqrtf((total_sum_sq / (float)hidden_dim) + eps);
    }
    __syncthreads();

    float r_scale = s_mean_sq;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        float w = __bfloat162float(Weight[i]);
        y_row[i] = __float2bfloat16(val * r_scale * w);
    }
}

extern "C" {

void launch_rmsnorm_forward(
    const __nv_bfloat16* X,
    const __nv_bfloat16* Weight,
    __nv_bfloat16* Y,
    int num_rows,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    int threads = (hidden_dim <= 512) ? 128 : 256;
    rmsnorm_forward_kernel<<<num_rows, threads, 0, stream>>>(
        X, Weight, Y, hidden_dim, eps
    );
}

}
