#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/**
 * Fused SwiGLU Native CUDA Kernel
 * 
 * Computes:
 * SwiGLU(gate, up) = (gate * sigmoid(gate)) * up
 * In-place in registers to avoid materializing intermediate activation tensors.
 */

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_forward_kernel(
    const __nv_bfloat16* __restrict__ Gate,
    const __nv_bfloat16* __restrict__ Up,
    __nv_bfloat16* __restrict__ Out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = __bfloat162float(Gate[idx]);
        float u = __bfloat162float(Up[idx]);
        float res = silu(g) * u;
        Out[idx] = __float2bfloat16(res);
    }
}

extern "C" {

void launch_swiglu_forward(
    const __nv_bfloat16* Gate,
    const __nv_bfloat16* Up,
    __nv_bfloat16* Out,
    int total_elements,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    swiglu_forward_kernel<<<blocks, threads, 0, stream>>>(
        Gate, Up, Out, total_elements
    );
}

}
