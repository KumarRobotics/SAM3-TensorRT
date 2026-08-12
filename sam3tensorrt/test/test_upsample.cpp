#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "sam3tensorrt/cuda/upsample.cuh"

// small test: 4x4 --> 16x16
//
// With a constant source, bilinear interpolation must reproduce that constant
// everywhere regardless of weights or border clamping:
//   result = c * (wy0*wx0 + wy0*wx1 + wy1*wx0 + wy1*wx1)
//          = c * (wy0+wy1) * (wx0+wx1)
//          = c * 1 * 1 = c  ✓

TEST(UpsampleMasksTest, SmallConstantFill) {
    // Full decoder output has 200 slots even if we only upsample a few
    constexpr int NUM_SLOTS = 200;
    constexpr int N = 3;
    constexpr int SRC_H = 4,  SRC_W = 4;
    constexpr int DST_H = 16, DST_W = 16;
    constexpr float FILL = 7.5f;

    const std::vector<int> indices_h = {0, 5, 10};

    // Fill only the selected slots with FILL, leave others at 0
    std::vector<float> masks_h(NUM_SLOTS * SRC_H * SRC_W, 0.0f);
    for (int idx : indices_h) {
        for (int i = 0; i < SRC_H * SRC_W; ++i) {
            masks_h[idx * SRC_H * SRC_W + i] = FILL;
        }
    }

    float* d_masks = nullptr;
    int* d_indices = nullptr;
    float* d_out = nullptr;

    ASSERT_EQ(cudaMalloc(&d_masks, NUM_SLOTS * SRC_H * SRC_W * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_indices, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, N * DST_H * DST_W * sizeof(float)), cudaSuccess);

    cudaMemcpy(d_masks, masks_h.data(), NUM_SLOTS * SRC_H * SRC_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices_h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    upsampleMasks(d_masks, d_indices, d_out, N, SRC_H, SRC_W, DST_H, DST_W, nullptr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> out_h(N * DST_H * DST_W);
    ASSERT_EQ(cudaMemcpy(out_h.data(), d_out, N * DST_H * DST_W * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N * DST_H * DST_W; ++i)
        EXPECT_FLOAT_EQ(out_h[i], FILL)
            << "output[" << i << "] expected " << FILL
            << " (mask " << i / (DST_H * DST_W)
            << ", pixel " << i % (DST_H * DST_W) << ")";

    cudaFree(d_masks);
    cudaFree(d_indices);
    cudaFree(d_out);
}

// larget test: ensure gpu can handle image size
TEST(UpsampleMasksTest, FullSizeGpuMemory) {
    constexpr int NUM_SLOTS = 200;
    constexpr int N = 200;
    constexpr int SRC_H = 184,  SRC_W = 184;
    constexpr int DST_H = 1536, DST_W = 2048;
    constexpr size_t OUT_BYTES = static_cast<size_t>(N) * DST_H * DST_W * sizeof(float); 

    std::vector<int> indices_h(N);
    std::iota(indices_h.begin(), indices_h.end(), 0);  // 0..199

    float* d_masks = nullptr;
    int* d_indices = nullptr;
    float* d_out = nullptr;

    // If the GPU doesn't have enough memory the test fails cleanly here
    ASSERT_EQ(cudaMalloc(&d_masks, NUM_SLOTS * SRC_H * SRC_W * sizeof(float)), cudaSuccess)
        << "Not enough GPU memory for source masks";
    ASSERT_EQ(cudaMalloc(&d_indices, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, OUT_BYTES), cudaSuccess)
        << "Not enough GPU memory for output (~2.4 GB). "
           "Reduce MAX_DST_H/W or the mask count.";

    // Use a non-zero fill so we can distinguish from unwritten memory
    cudaMemset(d_masks, 0x3f, NUM_SLOTS * SRC_H * SRC_W * sizeof(float));
    cudaMemcpy(d_indices, indices_h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    upsampleMasks(d_masks, d_indices, d_out, N, SRC_H, SRC_W, DST_H, DST_W, nullptr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel launch or execution failed";

    constexpr int SPOTS = 10;
    const size_t  total = static_cast<size_t>(N) * DST_H * DST_W;
    const size_t  step  = total / SPOTS;

    for (int s = 0; s < SPOTS; ++s) {
        float val = 0.0f;
        ASSERT_EQ(cudaMemcpy(&val, d_out + s * step, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::isfinite(val)) << "d_out[" << s * step << "] is not finite";
    }

    cudaFree(d_masks);
    cudaFree(d_indices);
    cudaFree(d_out);
}
