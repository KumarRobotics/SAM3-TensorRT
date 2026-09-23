#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * Fused BGR HWC uint8 → RGB NCHW float32 normalization.
 * Input is already resized to (H, W) — no spatial transform.
 *
 * @param d_bgr_hwc   Device ptr: [H, W, 3] BGR uint8
 * @param d_rgb_nchw  Device ptr: [1, 3, H, W] float32
 * @param H, W        Image dimensions (both 644 in practice)
 * @param mean        Per-channel mean   in RGB order
 * @param std_dev     Per-channel stddev in RGB order
 * @param stream      CUDA stream */
void normalizeImage(const uint8_t* d_bgr_hwc, float* d_rgb_nchw, int H, int W, const float mean[3], const float std_dev[3], cudaStream_t stream);
