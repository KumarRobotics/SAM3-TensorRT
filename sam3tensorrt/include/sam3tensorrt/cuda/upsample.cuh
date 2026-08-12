#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * Batched bilinear upsample: [N, src_h, src_w] -> [N, dst_h, dst_w]
 * Only upsamples surviving slots selected by d_indices.
 *
 * @param d_masks    Full decoder output [200, src_h, src_w] float32 — read only
 * @param d_indices  Indices of surviving slots [N] int32
 * @param d_out      Pre-allocated output [N, dst_h, dst_w] float32
 * @param N          Number of surviving masks
 * @param src_h      184
 * @param src_w      184
 * @param dst_h      Original image height (from original_sizes)
 * @param dst_w      Original image width  (from original_sizes)
 * @param stream     CUDA stream
 */
void upsampleMasks(const float* d_masks,
                   const int* d_indices,
                   float* d_out,
                   int N,
                   int src_h, int src_w,
                   int dst_h, int dst_w,
                   cudaStream_t stream);
