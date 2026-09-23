/**
 * @author Jason Hughes
 *
 *
*/

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

void preprocessFused(
    const uint8_t* d_bgr_hwc,   // [src_h, src_w, 3] BGR uint8
    float* d_rgb_nchw,  // [3, dst_h, dst_w] float32
    int src_h, int src_w,
    int dst_h, int dst_w,
    const float mean[3],        // RGB order
    const float std_dev[3],     // RGB order
    cudaStream_t stream = nullptr);
