// ane_model_compiler.m — High-level ANE compile functions callable from Swift
#import <Foundation/Foundation.h>
#include "ane_model_compiler.h"
#include "ane_mil_gen.h"

int ane_compiler_init(void) {
    return ane_bridge_init();
}

ANEKernelHandle *ane_compile_fused_gqa_qkv(
    const float *wq, const float *wk, const float *wv,
    int dim, int kv_dim, int spatial)
{
    @autoreleasepool {
        // Generate MIL text
        NSString *mil = mil_gen_gqa_qkv(dim, kv_dim, spatial);
        const char *milCStr = [mil UTF8String];
        size_t milLen = strlen(milCStr);

        // Build fused weight blob
        NSData *weightBlob = mil_build_gqa_qkv_weight_blob(wq, wk, wv, dim, kv_dim);

        // Input: x [1, dim, 1, spatial] fp32
        size_t inputSize = 1 * dim * 1 * spatial * sizeof(float);
        // Outputs: Q [1, dim, 1, spatial], K [1, kv_dim, 1, spatial], V [1, kv_dim, 1, spatial]
        size_t outputSizes[3] = {
            (size_t)(1 * dim * 1 * spatial * sizeof(float)),
            (size_t)(1 * kv_dim * 1 * spatial * sizeof(float)),
            (size_t)(1 * kv_dim * 1 * spatial * sizeof(float))
        };

        return ane_bridge_compile(
            milCStr, milLen,
            (const uint8_t *)[weightBlob bytes], [weightBlob length],
            1, &inputSize,
            3, outputSizes);
    }
}

ANEKernelHandle *ane_compile_conv(
    const float *weights, int in_ch, int out_ch, int spatial)
{
    @autoreleasepool {
        NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
        const char *milCStr = [mil UTF8String];
        size_t milLen = strlen(milCStr);

        NSData *weightBlob = mil_build_weight_blob(weights, out_ch, in_ch);

        size_t inputSize = 1 * in_ch * 1 * spatial * sizeof(float);
        size_t outputSize = 1 * out_ch * 1 * spatial * sizeof(float);

        return ane_bridge_compile(
            milCStr, milLen,
            (const uint8_t *)[weightBlob bytes], [weightBlob length],
            1, &inputSize,
            1, &outputSize);
    }
}

ANEKernelHandle *ane_compile_fused_ffn_up(
    const float *w1, const float *w3,
    int dim, int hidden_dim, int spatial)
{
    @autoreleasepool {
        NSString *mil = mil_gen_ffn_up(dim, hidden_dim, spatial);
        const char *milCStr = [mil UTF8String];
        size_t milLen = strlen(milCStr);

        NSData *weightBlob = mil_build_ffn_up_weight_blob(w1, w3, hidden_dim, dim);

        size_t inputSize = 1 * dim * 1 * spatial * sizeof(float);
        size_t outputSizes[2] = {
            (size_t)(1 * hidden_dim * 1 * spatial * sizeof(float)),
            (size_t)(1 * hidden_dim * 1 * spatial * sizeof(float))
        };

        return ane_bridge_compile(
            milCStr, milLen,
            (const uint8_t *)[weightBlob bytes], [weightBlob length],
            1, &inputSize,
            2, outputSizes);
    }
}

int ane_compiler_get_count(void) {
    return ane_bridge_get_compile_count();
}

void ane_compiler_free_kernel(ANEKernelHandle *kernel) {
    ane_bridge_free(kernel);
}
