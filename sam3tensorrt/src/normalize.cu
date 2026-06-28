#include "sam3tensorrt/cuda/normalize.cuh"

__global__ void normalizeKernel(const uint8_t* __restrict__ src, 
                                float* __restrict__ dst,
                                int H, 
                                int W,
                                float mean_r,
                                float mean_g,
                                float mean_b,
                                float inv_std_r,
                                float inv_std_g,
                                float inv_std_b)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // col
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // row
    if (x >= W || y >= H) return;

    const int src_idx = (y * W + x) * 3;
    const int pix_idx =  y * W + x;
    const int plane = H * W;

    const float inv255 = 1.0f / 255.0f;

    // BGR to RGB swap during load
    const float r = src[src_idx + 2] * inv255;
    const float g = src[src_idx + 1] * inv255;
    const float b = src[src_idx + 0] * inv255;

    // permute to NCHW... coalesced
    dst[0 * plane + pix_idx] = (r - mean_r) * inv_std_r;
    dst[1 * plane + pix_idx] = (g - mean_g) * inv_std_g;
    dst[2 * plane + pix_idx] = (b - mean_b) * inv_std_b;
}

void normalizeImage(const uint8_t* d_bgr_hwc, float* d_rgb_nchw, int H, int W, const float mean[3], const float std_dev[3], cudaStream_t stream)
{
    const dim3 block(32, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    normalizeKernel<<<grid, block, 0, stream>>>(d_bgr_hwc, d_rgb_nchw, H, W, mean[0], mean[1], mean[2], 1.0f / std_dev[0], 1.0f / std_dev[1], 1.0f / std_dev[2]);
}
