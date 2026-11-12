#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan.h>
#endif

/**
 * CUDA <-> Vulkan Zero-Copy Interop Engine
 *
 * Maps neuron activations and router decisions directly into Vulkan device memory / textures.
 * Enables 120 FPS real-time rendering in the desktop Studio using Vulkan / CUDA without CPU copies,
 * socket serialization, or host RAM allocation.
 */

// Colors: Reflex (Emerald Green), Limbic (Amber), Cortex (Purple/Cyan)
__device__ __forceinline__ uchar4 get_tier_color_vk(int tier, float intensity) {
    intensity = fminf(fmaxf(intensity, 0.0f), 1.0f);
    if (tier == 0) {
        // Reflex: Emerald Green (#10B981)
        return make_uchar4((unsigned char)(16 * intensity), (unsigned char)(185 * intensity), (unsigned char)(129 * intensity), 255);
    } else if (tier == 1) {
        // Limbic: Amber/Gold (#F59E0B)
        return make_uchar4((unsigned char)(245 * intensity), (unsigned char)(158 * intensity), (unsigned char)(11 * intensity), 255);
    } else {
        // Cortex: Electric Violet (#8B5CF6)
        return make_uchar4((unsigned char)(139 * intensity), (unsigned char)(92 * intensity), (unsigned char)(246 * intensity), 255);
    }
}

/**
 * Render Neuron Activation Matrix directly to Vulkan-mapped Linear Buffer
 */
__global__ void render_activations_to_vulkan_buffer_kernel(
    uchar4* __restrict__ vk_buffer,
    const __nv_bfloat16* __restrict__ activations,
    const int* __restrict__ selected_tiers,
    int width,
    int height,
    int hidden_dim
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Token / Sequence index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Layer / Hidden neuron index

    if (x < width && y < height) {
        int act_idx = x * hidden_dim + (y % hidden_dim);
        float val = fabsf(__bfloat162float(activations[act_idx]));
        int tier = selected_tiers[x % 3]; // Active tier for token

        uchar4 color = get_tier_color_vk(tier, val);
        vk_buffer[y * width + x] = color;
    }
}

extern "C" {

struct VulkanInteropContext {
    cudaExternalMemory_t ext_memory;
    void* dev_ptr;
    size_t size;
    int width;
    int height;
#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle;
#else
    int fd;
#endif
};

VulkanInteropContext* create_vulkan_interop_context_win32(
    void* handle,
    size_t total_bytes,
    int width,
    int height
) {
    VulkanInteropContext* ctx = new VulkanInteropContext();
    memset(ctx, 0, sizeof(VulkanInteropContext));
    ctx->size = total_bytes;
    ctx->width = width;
    ctx->height = height;

    cudaExternalMemoryHandleDesc memDesc;
    memset(&memDesc, 0, sizeof(memDesc));
#if defined(_WIN32) || defined(_WIN64)
    memDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memDesc.handle.win32.handle = (HANDLE)handle;
    ctx->handle = (HANDLE)handle;
#else
    memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memDesc.handle.fd = (int)(intptr_t)handle;
    ctx->fd = (int)(intptr_t)handle;
#endif
    memDesc.size = total_bytes;

    cudaError_t err = cudaImportExternalMemory(&ctx->ext_memory, &memDesc);
    if (err != cudaSuccess) {
        delete ctx;
        return nullptr;
    }

    cudaExternalMemoryBufferDesc bufDesc;
    memset(&bufDesc, 0, sizeof(bufDesc));
    bufDesc.offset = 0;
    bufDesc.size = total_bytes;

    err = cudaExternalMemoryGetMappedBuffer(&ctx->dev_ptr, ctx->ext_memory, &bufDesc);
    if (err != cudaSuccess) {
        cudaDestroyExternalMemory(ctx->ext_memory);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void render_neuron_activation_vulkan(
    VulkanInteropContext* ctx,
    const __nv_bfloat16* activations,
    const int* selected_tiers,
    int hidden_dim,
    cudaStream_t stream
) {
    if (!ctx || !ctx->dev_ptr) return;

    dim3 block(16, 16);
    dim3 grid((ctx->width + 15) / 16, (ctx->height + 15) / 16);
    render_activations_to_vulkan_buffer_kernel<<<grid, block, 0, stream>>>(
        (uchar4*)ctx->dev_ptr,
        activations,
        selected_tiers,
        ctx->width,
        ctx->height,
        hidden_dim
    );
}

void destroy_vulkan_interop_context(VulkanInteropContext* ctx) {
    if (ctx) {
        if (ctx->dev_ptr) {
            cudaFree(ctx->dev_ptr);
        }
        if (ctx->ext_memory) {
            cudaDestroyExternalMemory(ctx->ext_memory);
        }
        delete ctx;
    }
}

const char* get_graphics_interop_backend_name() {
    return "Vulkan / CUDA Zero-Copy Pipeline";
}

}
