#ifndef TRIUNE_ENGINE_API_H
#define TRIUNE_ENGINE_API_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef TRIUNE_EXPORTS
    #define TRIUNE_API __declspec(dllexport)
  #else
    #define TRIUNE_API __declspec(dllimport)
  #endif
#else
  #define TRIUNE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the native C++/CUDA Triune Engine instance
typedef struct TriuneEngine TriuneEngine;

// Minimalist Architecture Configuration
typedef struct {
    int vocab_size;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int head_dim;
    int num_experts;
    int top_k;
    int max_seq_len;
    float reflex_threshold;
} TriuneConfig;

// Real-time Telemetry Snapshot for Studio & CLI
typedef struct {
    float loss;
    float lm_loss;
    float router_loss;
    int active_tier;          // 0: Reflex, 1: Limbic, 2: Cortex
    int tokens_processed;
    float step_time_ms;
    float vram_used_mb;
    int cuda_runtime_bound;   // 1 if cudart64 / nvcuda bound dynamically
} TriuneStepStats;

// 1. Lifecycle Management
TRIUNE_API TriuneEngine* triune_engine_create(const TriuneConfig* config, int device_id);
TRIUNE_API void triune_engine_destroy(TriuneEngine* engine);
TRIUNE_API const char* triune_engine_version(void);
TRIUNE_API int triune_has_cuda_runtime(void);

// 2. Zero-Overhead Training & Inference Execution
TRIUNE_API int triune_engine_step(
    TriuneEngine* engine,
    const uint16_t* input_tokens,
    const uint16_t* target_tokens,
    int batch_size,
    int seq_len,
    TriuneStepStats* out_stats
);

// 3. CUDA Graph Capture & Replay
TRIUNE_API int triune_cuda_graph_capture(TriuneEngine* engine, int batch_size, int seq_len);
TRIUNE_API int triune_cuda_graph_replay(TriuneEngine* engine);

// 4. Native C/AVX2/CUDA Operator Entrypoints
TRIUNE_API void triune_centroid_steer_update_f32(
    float* P,
    const float* Grad_proj,
    int rows,
    int cols,
    float steer_scale,
    float momentum
);

TRIUNE_API void triune_route_tiers_f32(
    const float* logits,
    float* probs_out,
    int* selected_tiers_out,
    int batch_size,
    float reflex_threshold
);

TRIUNE_API void triune_compute_neuron_heatmap_f32(
    const float* activations,
    float* bins_out,
    uint32_t* rgba_out,
    int num_elements,
    int num_bins
);

TRIUNE_API void triune_gla_recurrent_step_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    float* state,
    float* out,
    int batch_heads,
    int head_dim
);

#ifdef __cplusplus
}
#endif

#endif // TRIUNE_ENGINE_API_H
