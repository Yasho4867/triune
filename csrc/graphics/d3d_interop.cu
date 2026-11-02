#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/**
 * CUDA ↔ Direct3D 11 Zero-Copy Interop Engine
 * 
 * Maps neuron activations and router decisions directly into Direct3D textures in VRAM.
 * Enables 120 FPS real-time rendering in the desktop Studio without CPU copies,
 * socket serialization, or host RAM allocation.
 */

// Colors: Reflex (Emerald Green), Limbic (Amber), Cortex (Purple/Cyan)
__device__ __forceinline__ uchar4 get_tier_color(int tier, float intensity) {
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
 * Render Neuron Activation Matrix to 2D Surface Kernel
 */
__global__ void render_activations_to_surface_kernel(
    cudaSurfaceObject_t surface,
    const __nv_bfloat16* __restrict__ Activations,
    const int* __restrict__ SelectedTiers,
    int width,
    int height,
    int hidden_dim
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Token / Sequence index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Layer / Hidden neuron index

    if (x < width && y < height) {
        int act_idx = x * hidden_dim + (y % hidden_dim);
        float val = fabsf(__bfloat162float(Activations[act_idx]));
        int tier = SelectedTiers[x % 3]; // Active tier for token

        uchar4 color = get_tier_color(tier, val);
        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    }
}

extern "C" {

struct D3DInteropContext {
    cudaGraphicsResource_t cuda_resource;
    ID3D11Texture2D* d3d_texture;
    int width;
    int height;
};

D3DInteropContext* create_d3d_interop_context(ID3D11Texture2D* texture, int width, int height) {
    D3DInteropContext* ctx = new D3DInteropContext();
    ctx->d3d_texture = texture;
    ctx->width = width;
    ctx->height = height;

    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &ctx->cuda_resource,
        texture,
        cudaGraphicsRegisterFlagsNone
    );
    if (err != cudaSuccess) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void render_neuron_activation_texture(
    D3DInteropContext* ctx,
    const __nv_bfloat16* activations,
    const int* selected_tiers,
    int hidden_dim,
    cudaStream_t stream
) {
    if (!ctx || !ctx->cuda_resource) return;

    // Map DirectX 11 texture into CUDA address space
    cudaGraphicsMapResources(1, &ctx->cuda_resource, stream);
    cudaArray_t cu_array;
    cudaGraphicsSubResourceGetMappedArray(&cu_array, ctx->cuda_resource, 0, 0);

    // Create CUDA surface object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array;

    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);

    dim3 block(16, 16);
    dim3 grid((ctx->width + 15) / 16, (ctx->height + 15) / 16);
    render_activations_to_surface_kernel<<<grid, block, 0, stream>>>(
        surface, activations, selected_tiers, ctx->width, ctx->height, hidden_dim
    );

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &ctx->cuda_resource, stream);
}

void destroy_d3d_interop_context(D3DInteropContext* ctx) {
    if (ctx) {
        if (ctx->cuda_resource) {
            cudaGraphicsUnregisterResource(ctx->cuda_resource);
        }
        delete ctx;
    }
}

}
