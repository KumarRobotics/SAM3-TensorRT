#include "sam3tensorrt/cuda/preprocess.cuh"

__global__ void resizeNormFusedKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale_y, float scale_x,          // src_h/dst_h, src_w/dst_w
    float mean_r, float mean_g, float mean_b,
    float inv_std_r, float inv_std_g, float inv_std_b)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    if (ox >= dst_w || oy >= dst_h) return;

    const float iy = (oy + 0.5f) * scale_y - 0.5f;
    const float ix = (ox + 0.5f) * scale_x - 0.5f;

    const int   y0f = (int)floorf(iy);
    const int   x0f = (int)floorf(ix);
    const float wy1 = iy - (float)y0f;   
    const float wy0 = 1.0f - wy1;
    const float wx1 = ix - (float)x0f;   
    const float wx0 = 1.0f - wx1;

    const int y0 = max(0, min(y0f, src_h - 1));
    const int y1 = max(0, min(y0f + 1, src_h - 1));
    const int x0 = max(0, min(x0f, src_w - 1));
    const int x1 = max(0, min(x0f + 1, src_w - 1));

    const uint8_t* p00 = src + (y0 * src_w + x0) * 3;
    const uint8_t* p01 = src + (y0 * src_w + x1) * 3;
    const uint8_t* p10 = src + (y1 * src_w + x0) * 3;
    const uint8_t* p11 = src + (y1 * src_w + x1) * 3;

    const float inv255 = 1.0f / 255.0f;
    const float r = (p00[2]*wx0*wy0 + p01[2]*wx1*wy0 + p10[2]*wx0*wy1 + p11[2]*wx1*wy1) * inv255;
    const float g = (p00[1]*wx0*wy0 + p01[1]*wx1*wy0 + p10[1]*wx0*wy1 + p11[1]*wx1*wy1) * inv255;
    const float b = (p00[0]*wx0*wy0 + p01[0]*wx1*wy0 + p10[0]*wx0*wy1 + p11[0]*wx1*wy1) * inv255;

    const int plane   = dst_h * dst_w;
    const int pix_idx = oy * dst_w + ox;
    dst[0 * plane + pix_idx] = (r - mean_r) * inv_std_r;
    dst[1 * plane + pix_idx] = (g - mean_g) * inv_std_g;
    dst[2 * plane + pix_idx] = (b - mean_b) * inv_std_b;
}

void preprocessFused(
    const uint8_t* d_bgr_hwc, float* d_rgb_nchw,
    int src_h, int src_w, int dst_h, int dst_w,
    const float mean[3], const float std_dev[3],
    cudaStream_t stream)
{
    const dim3 block(32, 16);
    const dim3 grid((dst_w + block.x-1) / block.x,
                    (dst_h + block.y-1) / block.y);

    resizeNormFusedKernel<<<grid, block, 0, stream>>>(
        d_bgr_hwc, d_rgb_nchw,
        src_h, src_w, dst_h, dst_w,
        (float)src_h / dst_h,       // scale_y
        (float)src_w / dst_w,       // scale_x
        mean[0], mean[1], mean[2],
        1.0f/std_dev[0], 1.0f/std_dev[1], 1.0f/std_dev[2]);
}
