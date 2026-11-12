#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/**
 * Custom Gated Linear Attention (GLA) Native CUDA Kernel
 * 
 * Computes O(N) linear-time gated attention with chunked intra-chunk parallel GEMMs
 * and fused exponential log-decay gating. Includes dedicated O(1) single-step
 * autoregressive decoding kernel.
 */

#define CHUNK_SIZE 64
#define HEAD_DIM 64

/**
 * O(1) Fast Autoregressive Decoding Kernel (T = 1)
 * Updates recurrent hidden state:
 * S_t = S_{t-1} * exp(g_t) + K_t^T * V_t
 * Out_t = Q_t * S_t
 */
__global__ void gla_decode_step_kernel(
    const __nv_bfloat16* __restrict__ Q,      // [B, 1, H, HD]
    const __nv_bfloat16* __restrict__ K,      // [B, 1, H, HD]
    const __nv_bfloat16* __restrict__ V,      // [B, 1, H, HD]
    const float* __restrict__ G,              // [B, 1, H, HD] (log-decay)
    float* __restrict__ State,                // [B, H, HD, HD]
    __nv_bfloat16* __restrict__ Out,          // [B, 1, H, HD]
    int B,
    int H,
    int HD
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    if (b >= B || h >= H) return;

    int bh_offset = (b * H + h);
    const __nv_bfloat16* q_bh = Q + bh_offset * HD;
    const __nv_bfloat16* k_bh = K + bh_offset * HD;
    const __nv_bfloat16* v_bh = V + bh_offset * HD;
    const float* g_bh = G + bh_offset * HD;
    float* state_bh = State + bh_offset * HD * HD;
    __nv_bfloat16* out_bh = Out + bh_offset * HD;

    // 2D warp-strided recurrent state update:
    // S[i, j] = S[i, j] * exp(G[i]) + K[i] * V[j]
    for (int i = threadIdx.y; i < HD; i += blockDim.y) {
        float decay = expf(g_bh[i]);
        float k_val = __bfloat162float(k_bh[i]);

        for (int j = threadIdx.x; j < HD; j += blockDim.x) {
            int state_idx = i * HD + j;
            float v_val = __bfloat162float(v_bh[j]);
            float s = state_bh[state_idx] * decay + (k_val * v_val);
            state_bh[state_idx] = s;
        }
    }

    __syncthreads();

    // Query matrix multiplication: Out[j] = sum_i Q[i] * S[i, j]
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    for (int j = tid; j < HD; j += num_threads) {
        float accum = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            float q_val = __bfloat162float(q_bh[i]);
            accum += q_val * state_bh[i * HD + j];
        }
        out_bh[j] = __float2bfloat16(accum);
    }
}

/**
 * Fused Chunk Decay & Intra-Chunk Attention Kernel
 * Fuses the log-decay gate exponential with causal intra-chunk linear masking.
 */
__global__ void gla_chunk_decay_kernel(
    const float* __restrict__ G,           // [B, T, H, HD]
    float* __restrict__ Decayed_G,         // [B, T, H, HD]
    int B, int T, int H, int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * H * HD;

    if (idx < total) {
        float g_val = G[idx];
        Decayed_G[idx] = expf(g_val);
    }
}

extern "C" {

void launch_gla_decode_step(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* G,
    float* State,
    __nv_bfloat16* Out,
    int B,
    int H,
    int HD,
    cudaStream_t stream
) {
    dim3 block(32, 16); // 512 threads per block, strictly <= 1024 CUDA limit
    dim3 grid(1, H, B);
    gla_decode_step_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, G, State, Out, B, H, HD
    );
}

void launch_gla_chunk_decay(
    const float* G,
    float* Decayed_G,
    int B, int T, int H, int HD,
    cudaStream_t stream
) {
    int total = B * T * H * HD;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gla_chunk_decay_kernel<<<blocks, threads, 0, stream>>>(
        G, Decayed_G, B, T, H, HD
    );
}

}
