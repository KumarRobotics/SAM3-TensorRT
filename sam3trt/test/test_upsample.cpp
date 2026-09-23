#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>

#include "sam3trt/cuda/upsample.cuh"

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

static std::vector<float> runUpsample(const std::vector<float>& masks_h, const std::vector<int>& indices_h,
                                      int src_h, int src_w, int dst_h, int dst_w) {
    const int n = static_cast<int>(indices_h.size());
    const size_t out_count = static_cast<size_t>(n) * dst_h * dst_w;

    float* d_masks = nullptr;
    int* d_indices = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_masks, masks_h.size() * sizeof(float));
    cudaMalloc(&d_indices, n * sizeof(int));
    cudaMalloc(&d_out, out_count * sizeof(float));
    cudaMemcpy(d_masks, masks_h.data(), masks_h.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices_h.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    upsampleMasks(d_masks, d_indices, d_out, n, src_h, src_w, dst_h, dst_w, nullptr);

    std::vector<float> out_h(out_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(out_h.data(), d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_masks);
    cudaFree(d_indices);
    cudaFree(d_out);
    return out_h;
}

TEST(UpsampleMasksTest, SelectsIndexedSlots) {
    constexpr int NUM_SLOTS = 200;
    constexpr int SRC = 8, DST = 20;

    std::vector<float> masks_h(NUM_SLOTS * SRC * SRC);
    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        std::fill_n(masks_h.begin() + slot * SRC * SRC, SRC * SRC, static_cast<float>(slot));
    }

    const std::vector<int> indices_h = {199, 3, 42};
    const std::vector<float> out_h = runUpsample(masks_h, indices_h, SRC, SRC, DST, DST);

    for (size_t m = 0; m < indices_h.size(); ++m) {
        for (int i = 0; i < DST * DST; ++i) {
            ASSERT_FLOAT_EQ(out_h[m * DST * DST + i], static_cast<float>(indices_h[m])) << "mask " << m;
        }
    }
}

TEST(UpsampleMasksTest, MatchesOpenCvBilinear) {
    constexpr int NUM_SLOTS = 4;
    constexpr int SRC_H = 184, SRC_W = 184;
    constexpr int DST_H = 375, DST_W = 500;

    std::vector<float> masks_h(NUM_SLOTS * SRC_H * SRC_W);
    cv::Mat random(1, static_cast<int>(masks_h.size()), CV_32FC1, masks_h.data());
    cv::randn(random, 0.0, 1.0);

    const std::vector<int> indices_h = {2, 0};
    const std::vector<float> out_h = runUpsample(masks_h, indices_h, SRC_H, SRC_W, DST_H, DST_W);

    for (size_t m = 0; m < indices_h.size(); ++m) {
        cv::Mat src(SRC_H, SRC_W, CV_32FC1, masks_h.data() + indices_h[m] * SRC_H * SRC_W);
        cv::Mat expected;
        cv::resize(src, expected, cv::Size(DST_W, DST_H), 0, 0, cv::INTER_LINEAR);

        cv::Mat result(DST_H, DST_W, CV_32FC1, const_cast<float*>(out_h.data()) + m * DST_H * DST_W);
        EXPECT_LT(cv::norm(result, expected, cv::NORM_INF), 1e-3) << "mask " << m;
    }
}
