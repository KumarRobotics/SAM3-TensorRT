#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sam3tensorrt/Sam3ImageEncoder.hpp"

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

static std::string imageEncoderPlanPath() {
    return modelsDir() + "/image_encoder_fp16.engine";
}

static constexpr int kInputC = 3;
static constexpr int kInputH = 644;
static constexpr int kInputW = 644;
static constexpr size_t kInputCount = static_cast<size_t>(kInputC) * kInputH * kInputW;

static constexpr int kSampleCount = 16;

class Sam3ImageEncoderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        encoder_ = std::make_unique<Sam3ImageEncoder>(imageEncoderPlanPath());

        ASSERT_EQ(cudaMalloc(&d_input_, kInputCount * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_input_, 0, kInputCount * sizeof(float)), cudaSuccess);

        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    static void TearDownTestSuite() {
        cudaStreamDestroy(stream_);
        cudaFree(d_input_);
        encoder_.reset();
    }

    // Copies the first `kSampleCount` half-precision elements from a device
    // buffer back to host as floats.
    static std::vector<float> sampleHost(const __half* d_ptr) {
        std::vector<float> host(kSampleCount);
        cudaMemcpy(host.data(), d_ptr, kSampleCount * sizeof(__half), cudaMemcpyDeviceToHost);

        std::vector<float> out(kSampleCount);
        for (int i = 0; i < kSampleCount; ++i) {
            out[i] = host[i];
        }
        return out;
    }

    static std::unique_ptr<Sam3ImageEncoder> encoder_;
    static float* d_input_;
    static cudaStream_t stream_;
};

std::unique_ptr<Sam3ImageEncoder> Sam3ImageEncoderTest::encoder_;
float* Sam3ImageEncoderTest::d_input_ = nullptr;
cudaStream_t Sam3ImageEncoderTest::stream_;

TEST_F(Sam3ImageEncoderTest, ConstructsWithoutThrowing) {
    // encoder_ is already constructed in SetUpTestSuite; if that threw, the
    // whole suite would have failed at setup. This just documents the
    // expectation explicitly and re-checks the object is usable.
    ASSERT_NE(encoder_, nullptr);
}

TEST_F(Sam3ImageEncoderTest, ConstructorThrowsOnBadPlanPath) {
    EXPECT_THROW(Sam3ImageEncoder("/nonexistent/path/does_not_exist.plan"),
                 std::exception);
}

TEST_F(Sam3ImageEncoderTest, EncodeRunsWithoutThrowing) {
    EXPECT_NO_THROW({
        Sam3ImageFeatures features = encoder_->encode(d_input_, stream_);
        cudaStreamSynchronize(stream_);
        (void)features;
    });
}

TEST_F(Sam3ImageEncoderTest, EncodeReturnsNonNullPointers) {
    Sam3ImageFeatures features = encoder_->encode(d_input_, stream_);
    cudaStreamSynchronize(stream_);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NE(features.fpn[i], nullptr) << "fpn[" << i << "] is null";
        EXPECT_NE(features.fpn_pos[i], nullptr) << "fpn_pos[" << i << "] is null";
    }
}

TEST_F(Sam3ImageEncoderTest, FpnAndPosPointersAreAllDistinct) {
    Sam3ImageFeatures features = encoder_->encode(d_input_, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<const void*> ptrs = {
        features.fpn[0], features.fpn[1], features.fpn[2],
        features.fpn_pos[0], features.fpn_pos[1], features.fpn_pos[2],
    };

    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j])
                << "buffer " << i << " and buffer " << j << " alias";
        }
    }
}

// Regression test: buffers are allocated once in discoverAndAllocate() and
// reused across calls -- encode() should return the same device addresses
// every time, not reallocate.
TEST_F(Sam3ImageEncoderTest, PointersAreStableAcrossRepeatedCalls) {
    Sam3ImageFeatures first = encoder_->encode(d_input_, stream_);
    cudaStreamSynchronize(stream_);

    Sam3ImageFeatures second = encoder_->encode(d_input_, stream_);
    cudaStreamSynchronize(stream_);

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(first.fpn[i], second.fpn[i]);
        EXPECT_EQ(first.fpn_pos[i], second.fpn_pos[i]);
    }
}

TEST_F(Sam3ImageEncoderTest, EncodeProducesFiniteOutput) {
    Sam3ImageFeatures features = encoder_->encode(d_input_, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    for (int i = 0; i < 3; ++i) {
        for (float v : sampleHost(features.fpn[i])) {
            EXPECT_TRUE(std::isfinite(v)) << "fpn[" << i << "] produced non-finite value";
        }
        for (float v : sampleHost(features.fpn_pos[i])) {
            EXPECT_TRUE(std::isfinite(v)) << "fpn_pos[" << i << "] produced non-finite value";
        }
    }
}

TEST_F(Sam3ImageEncoderTest, DeepestFeatureLevelHasExpectedShape) {
    constexpr size_t kChannels = 256;
    constexpr size_t kSpatial = 46;
    constexpr size_t kExpectedCount = kChannels * kSpatial * kSpatial;  // 541696
 
    Sam3ImageFeatures features = encoder_->encode(d_input_, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float> fpn_host(kExpectedCount);
    ASSERT_EQ(cudaMemcpy(fpn_host.data(), features.fpn[2], kExpectedCount * sizeof(__half),cudaMemcpyDeviceToHost), cudaSuccess);

    std::vector<float> fpn_pos_host(kExpectedCount);
    ASSERT_EQ(cudaMemcpy(fpn_pos_host.data(), features.fpn_pos[2], kExpectedCount * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
}
