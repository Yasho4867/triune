#ifndef TRIUNE_ENGINE_API_H
#define TRIUNE_ENGINE_API_H

#ifdef _WIN32
  #ifdef TRIUNE_EXPORTS
    #define TRIUNE_API __declspec(dllexport)
  #else
    #define TRIUNE_API __declspec(dllimport)
  #endif
#else
  #define TRIUNE_API __attribute__((visibility("default")))
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TriuneEngine TriuneEngine;

typedef struct {
    int num_layers;
    int hidden_dim;
    int num_heads;
    int vocab_size;
    int max_seq_len;
    float reflex_threshold;
    float steer_scale;
    int galore_rank;
} TriuneConfig;

typedef struct {
    float loss;
    float lm_loss;
    float router_loss;
    int active_tier;
    int tokens_processed;
    float step_time_ms;
} TriuneStepStats;

// Core Engine Lifecycle
TRIUNE_API TriuneEngine* triune_engine_create(const TriuneConfig* config, int device_id);
TRIUNE_API void triune_engine_destroy(TriuneEngine* engine);

// Hardware Optimization Execution
TRIUNE_API int triune_engine_step(
    TriuneEngine* engine,
    const uint16_t* input_tokens,
    const uint16_t* target_tokens,
    int batch_size,
    int seq_len,
    TriuneStepStats* out_stats
);

// CentroidSteer Optimization Functions
TRIUNE_API void triune_centroid_steer_update(
    TriuneEngine* engine,
    float steer_scale,
    float momentum
);

// Custom GLA Attention Inference
TRIUNE_API void triune_gla_forward(
    TriuneEngine* engine,
    const void* q, const void* k, const void* v, const void* g,
    void* out, void* state,
    int B, int T, int H, int HD
);

// CUDA Graph Capture & Replay
TRIUNE_API int triune_cuda_graph_capture(TriuneEngine* engine, int batch_size, int seq_len);
TRIUNE_API int triune_cuda_graph_replay(TriuneEngine* engine);

// Direct3D 11 Texture Sharing for Neuron Visualizer
TRIUNE_API void* triune_d3d_create_interop(void* d3d_texture, int width, int height);
TRIUNE_API void triune_d3d_render_neurons(void* interop_ctx, TriuneEngine* engine);
TRIUNE_API void triune_d3d_destroy_interop(void* interop_ctx);

#ifdef __cplusplus
}
#endif

#endif // TRIUNE_ENGINE_API_H
