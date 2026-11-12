#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/**
 * 3-Tier Dynamic Router (Reflex / Limbic / Cortex) Native CUDA Kernel
 * 
 * Computes warp-level routing gates and early-exit thresholds directly on-chip.
 * If token/batch router confidence satisfies the early-exit criterion, execution
 * terminates immediately at Layer 2 (Reflex) or Layer 5 (Limbic) with ZERO CPU latency.
 */

#define WARP_SIZE 32

__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void init_early_exit_flag_kernel(int* GlobalEarlyExitFlag, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *GlobalEarlyExitFlag = (B > 0) ? 1 : 0;
    }
}

/**
 * Fused Router Gating & Warp-Level Early-Exit Evaluation
 * Tier 0: Reflex (Early exit at Layer 2)
 * Tier 1: Limbic (Intermediate exit at Layer 5)
 * Tier 2: Cortex (Deep execution through Layer 8)
 */
__global__ void evaluate_triune_routing_kernel(
    const float* __restrict__ RouterLogits,    // [B, 3] (Logits for Reflex, Limbic, Cortex)
    float* __restrict__ OutProbs,              // [B, 3] (Softmax probabilities)
    int* __restrict__ SelectedTiers,           // [B]    (Selected tier 0, 1, or 2)
    int* __restrict__ GlobalEarlyExitFlag,     // [1]    (Flag: 1 if all tokens early exit)
    float reflex_threshold,
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // Read 3 tier logits
    float l0 = RouterLogits[b * 3 + 0]; // Reflex
    float l1 = RouterLogits[b * 3 + 1]; // Limbic
    float l2 = RouterLogits[b * 3 + 2]; // Cortex

    // Softmax calculation
    float max_l = fmaxf(l0, fmaxf(l1, l2));
    float e0 = expf(l0 - max_l);
    float e1 = expf(l1 - max_l);
    float e2 = expf(l2 - max_l);
    float sum_e = e0 + e1 + e2;

    float p0 = e0 / sum_e;
    float p1 = e1 / sum_e;
    float p2 = e2 / sum_e;

    OutProbs[b * 3 + 0] = p0;
    OutProbs[b * 3 + 1] = p1;
    OutProbs[b * 3 + 2] = p2;

    // Determine tier choice (argmax or threshold)
    int choice = 2; // Default Cortex
    if (p0 >= reflex_threshold) {
        choice = 0; // Reflex
    } else if (p1 >= p2) {
        choice = 1; // Limbic
    }
    SelectedTiers[b] = choice;

    // Active warp-level vote: if any token in this warp is not Reflex, clear global flag
    unsigned int active = __activemask();
    unsigned int reflex_mask = __ballot_sync(active, choice == 0);
    if (reflex_mask != active) {
        atomicExch(GlobalEarlyExitFlag, 0);
    }
}

extern "C" {

void launch_triune_router(
    const float* RouterLogits,
    float* OutProbs,
    int* SelectedTiers,
    int* GlobalEarlyExitFlag,
    float reflex_threshold,
    int B,
    cudaStream_t stream
) {
    init_early_exit_flag_kernel<<<1, 1, 0, stream>>>(GlobalEarlyExitFlag, B);
    int threads = 128;
    int blocks = (B + threads - 1) / threads;
    evaluate_triune_routing_kernel<<<blocks, threads, 0, stream>>>(
        RouterLogits, OutProbs, SelectedTiers, GlobalEarlyExitFlag, reflex_threshold, B
    );
}

}
