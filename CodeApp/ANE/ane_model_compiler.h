// ane_model_compiler.h — High-level ANE compile functions callable from Swift
#ifndef ANE_MODEL_COMPILER_H
#define ANE_MODEL_COMPILER_H

#include "ane_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize ANE runtime (wraps ane_bridge_init)
int ane_compiler_init(void);

// Compile a fused GQA QKV kernel (Q: dim→dim, K: dim→kv_dim, V: dim→kv_dim)
// Returns kernel handle with 1 input (x) and 3 outputs (Q, K, V)
ANEKernelHandle *ane_compile_fused_gqa_qkv(
    const float *wq, const float *wk, const float *wv,
    int dim, int kv_dim, int spatial);

// Compile a single conv kernel (output projection or FFN down)
// Returns kernel handle with 1 input and 1 output
ANEKernelHandle *ane_compile_conv(
    const float *weights, int in_ch, int out_ch, int spatial);

// Compile fused FFN up kernel (W1 + W3 parallel convs for SwiGLU)
// Returns kernel handle with 1 input and 2 outputs (h1, h3)
ANEKernelHandle *ane_compile_fused_ffn_up(
    const float *w1, const float *w3,
    int dim, int hidden_dim, int spatial);

// Get current compile count
int ane_compiler_get_count(void);

// Free a kernel
void ane_compiler_free_kernel(ANEKernelHandle *kernel);

#ifdef __cplusplus
}
#endif

#endif // ANE_MODEL_COMPILER_H
