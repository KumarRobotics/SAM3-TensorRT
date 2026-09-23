#include "sam3trt/cuda/upsample.cuh"

__global__ void upsampleMasksKernel(
    const float* __restrict__ masks,    // [200, src_h, src_w]
    const int* __restrict__ indices,  // [N]
    float* __restrict__ out,      // [N, dst_h, dst_w]
    int N,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale_y,   // src_h / dst_h
    float scale_x)   // src_w / dst_w
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    const int m  = blockIdx.z;  // mask index in [0, N)

    if (ox >= dst_w || oy >= dst_h || m >= N) return;

    // matche torch align_corners=False
    const float iy = (oy + 0.5f) * scale_y - 0.5f;
    const float ix = (ox + 0.5f) * scale_x - 0.5f;

    const int y0f = static_cast<int>(floorf(iy));
    const int x0f = static_cast<int>(floorf(ix));

    const float wy1 = iy - static_cast<float>(y0f);
    const float wx1 = ix - static_cast<float>(x0f);
    const float wy0 = 1.0f - wy1;
    const float wx0 = 1.0f - wx1;

    const int y0 = max(0, min(y0f,     src_h - 1));
    const int y1 = max(0, min(y0f + 1, src_h - 1));
    const int x0 = max(0, min(x0f,     src_w - 1));
    const int x1 = max(0, min(x0f + 1, src_w - 1));

    // source slice for this surviving slot
    const int slot = indices[m];
    const float* src = masks + slot * src_h * src_w;

    const float val =
        src[y0 * src_w + x0] * wy0 * wx0 +
        src[y0 * src_w + x1] * wy0 * wx1 +
        src[y1 * src_w + x0] * wy1 * wx0 +
        src[y1 * src_w + x1] * wy1 * wx1;

    out[m * dst_h * dst_w + oy * dst_w + ox] = val;
}

void upsampleMasks(const float* d_masks,
                   const int* d_indices,
                   float* d_out,
                   int N,
                   int src_h, int src_w,
                   int dst_h, int dst_w,
                   cudaStream_t stream)
{
    if (N == 0) return;

    const dim3 block(32, 16, 1);
    const dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y,
        N);

    upsampleMasksKernel<<<grid, block, 0, stream>>>(
        d_masks, d_indices, d_out,
        N, src_h, src_w, dst_h, dst_w,
        static_cast<float>(src_h) / static_cast<float>(dst_h),
        static_cast<float>(src_w) / static_cast<float>(dst_w));
}
