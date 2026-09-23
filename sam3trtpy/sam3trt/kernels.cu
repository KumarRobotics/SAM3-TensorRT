#include <cuda_fp16.h>

template <typename T>
__device__ void resizeNormalize(
    const unsigned char* __restrict__ src,
    T* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float mean_r, float mean_g, float mean_b,
    float inv_std_r, float inv_std_g, float inv_std_b)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    const float iy = (oy + 0.5f) * ((float)src_h / dst_h) - 0.5f;
    const float ix = (ox + 0.5f) * ((float)src_w / dst_w) - 0.5f;

    const int y0f = (int)floorf(iy);
    const int x0f = (int)floorf(ix);
    const float wy1 = iy - (float)y0f;
    const float wy0 = 1.0f - wy1;
    const float wx1 = ix - (float)x0f;
    const float wx0 = 1.0f - wx1;

    const int y0 = max(0, min(y0f, src_h - 1));
    const int y1 = max(0, min(y0f + 1, src_h - 1));
    const int x0 = max(0, min(x0f, src_w - 1));
    const int x1 = max(0, min(x0f + 1, src_w - 1));

    const unsigned char* p00 = src + (y0 * src_w + x0) * 3;
    const unsigned char* p01 = src + (y0 * src_w + x1) * 3;
    const unsigned char* p10 = src + (y1 * src_w + x0) * 3;
    const unsigned char* p11 = src + (y1 * src_w + x1) * 3;

    const float inv255 = 1.0f / 255.0f;
    const float r = (p00[2]*wx0*wy0 + p01[2]*wx1*wy0 + p10[2]*wx0*wy1 + p11[2]*wx1*wy1) * inv255;
    const float g = (p00[1]*wx0*wy0 + p01[1]*wx1*wy0 + p10[1]*wx0*wy1 + p11[1]*wx1*wy1) * inv255;
    const float b = (p00[0]*wx0*wy0 + p01[0]*wx1*wy0 + p10[0]*wx0*wy1 + p11[0]*wx1*wy1) * inv255;

    const int plane = dst_h * dst_w;
    const int pix_idx = oy * dst_w + ox;
    dst[0 * plane + pix_idx] = T((r - mean_r) * inv_std_r);
    dst[1 * plane + pix_idx] = T((g - mean_g) * inv_std_g);
    dst[2 * plane + pix_idx] = T((b - mean_b) * inv_std_b);
}

#define RESIZE_NORMALIZE(NAME, T)                                                    \
extern "C" __global__ void NAME(                                                     \
    const unsigned char* src, T* dst, int src_h, int src_w, int dst_h, int dst_w,    \
    float mean_r, float mean_g, float mean_b,                                        \
    float inv_std_r, float inv_std_g, float inv_std_b)                               \
{                                                                                    \
    resizeNormalize<T>(src, dst, src_h, src_w, dst_h, dst_w,                         \
                       mean_r, mean_g, mean_b, inv_std_r, inv_std_g, inv_std_b);     \
}

RESIZE_NORMALIZE(resizeNormalizeFloat, float)
RESIZE_NORMALIZE(resizeNormalizeHalf, __half)

extern "C" __global__ void upsampleMasks(
    const __half* __restrict__ masks,
    const int* __restrict__ indices,
    unsigned char* __restrict__ out,
    int* __restrict__ boxes,
    int src_h, int src_w,
    int dst_h, int dst_w)
{
    __shared__ int block_box[4];

    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int m = blockIdx.z;
    const bool first = threadIdx.x == 0 && threadIdx.y == 0;

    if (first) {
        block_box[0] = dst_w;
        block_box[1] = dst_h;
        block_box[2] = -1;
        block_box[3] = -1;
    }
    __syncthreads();

    if (ox < dst_w && oy < dst_h) {
        const float iy = (oy + 0.5f) * ((float)src_h / dst_h) - 0.5f;
        const float ix = (ox + 0.5f) * ((float)src_w / dst_w) - 0.5f;

        const int y0f = (int)floorf(iy);
        const int x0f = (int)floorf(ix);
        const float wy1 = iy - (float)y0f;
        const float wx1 = ix - (float)x0f;
        const float wy0 = 1.0f - wy1;
        const float wx0 = 1.0f - wx1;

        const int y0 = max(0, min(y0f, src_h - 1));
        const int y1 = max(0, min(y0f + 1, src_h - 1));
        const int x0 = max(0, min(x0f, src_w - 1));
        const int x1 = max(0, min(x0f + 1, src_w - 1));

        const __half* src = masks + (size_t)indices[m] * src_h * src_w;
        const float val =
            __half2float(src[y0 * src_w + x0]) * wy0 * wx0 +
            __half2float(src[y0 * src_w + x1]) * wy0 * wx1 +
            __half2float(src[y1 * src_w + x0]) * wy1 * wx0 +
            __half2float(src[y1 * src_w + x1]) * wy1 * wx1;

        const bool on = val > 0.0f;
        out[(size_t)m * dst_h * dst_w + oy * dst_w + ox] = on ? 255 : 0;

        if (on) {
            atomicMin(&block_box[0], ox);
            atomicMin(&block_box[1], oy);
            atomicMax(&block_box[2], ox);
            atomicMax(&block_box[3], oy);
        }
    }
    __syncthreads();

    if (first && block_box[2] >= 0) {
        int* box = boxes + m * 4;
        atomicMin(&box[0], block_box[0]);
        atomicMin(&box[1], block_box[1]);
        atomicMax(&box[2], block_box[2]);
        atomicMax(&box[3], block_box[3]);
    }
}
