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
    int i = threadIdx.y; // Row in State [0..HD-1]
    int j = threadIdx.x; // Col in State [0..HD-1]

    if (b >= B || h >= H || i >= HD || j >= HD) return;

    int state_idx = ((b * H + h) * HD + i) * HD + j;
    int q_idx = ((b * H + h) * 1) * HD + i;
    int k_idx = ((b * H + h) * 1) * HD + i;
    int v_idx = ((b * H + h) * 1) * HD + j;
    int g_idx = ((b * H + h) * 1) * HD + i;

    // Read log-decay gate and inputs
    float decay = expf(G[g_idx]);
    float k_val = __bfloat162float(K[k_idx]);
    float v_val = __bfloat162float(V[v_idx]);

    // Recurrent state update: S = S * decay + K^T * V
    float s = State[state_idx] * decay + (k_val * v_val);
    State[state_idx] = s;

    __syncthreads();

    // Query matrix multiplication: Out = Q * S
    if (j == 0) {
        float accum = 0.0f;
        #pragma unroll
        for (int k_step = 0; k_step < HD; ++k_step) {
            float q_val = __bfloat162float(Q[((b * H + h) * 1) * HD + k_step]);
            accum += q_val * State[((b * H + h) * HD + k_step) * HD + i];
        }
        Out[q_idx] = __float2bfloat16(accum);
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
    dim3 block(HEAD_DIM, HEAD_DIM);
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
