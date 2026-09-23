#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sam3trt/Sam3TextEncoder.hpp"

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

static std::string textEncoderPlanPath() {
    return modelsDir() + "/text_encoder_fp16.engine";
}

static constexpr int kTextLen = 32;

static constexpr size_t kTextFeaturesCount = 32 * 1 * 256;    // 8192
static constexpr size_t kTextEmbeddingsCount = 32 * 1 * 1024; // 32768

static constexpr int kSampleCount = 16;

class Sam3TextEncoderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        encoder_ = std::make_unique<Sam3TextEncoder>(textEncoderPlanPath());

        ASSERT_EQ(cudaMalloc(&d_input_ids_, kTextLen * sizeof(int64_t)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_input_ids_, 0, kTextLen * sizeof(int64_t)), cudaSuccess);

        ASSERT_EQ(cudaMalloc(&d_attention_mask_, kTextLen * sizeof(bool)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_attention_mask_, 0, kTextLen * sizeof(bool)), cudaSuccess);

        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    static void TearDownTestSuite() {
        cudaStreamDestroy(stream_);
        cudaFree(d_input_ids_);
        cudaFree(d_attention_mask_);
        encoder_.reset();
    }

    // Copies `n` fp16 elements from a device buffer back to host, converted to float.
    static std::vector<float> sampleHost(const __half* d_ptr, size_t n = kSampleCount) {
        std::vector<__half> host(n);
        cudaMemcpy(host.data(), d_ptr, n * sizeof(__half), cudaMemcpyDeviceToHost);

        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = __half2float(host[i]);
        }
        return out;
    }

    static std::unique_ptr<Sam3TextEncoder> encoder_;
    static int64_t*                         d_input_ids_;
    static bool*                            d_attention_mask_;
    static cudaStream_t                     stream_;
};

std::unique_ptr<Sam3TextEncoder> Sam3TextEncoderTest::encoder_;
int64_t* Sam3TextEncoderTest::d_input_ids_ = nullptr;
bool* Sam3TextEncoderTest::d_attention_mask_ = nullptr;
cudaStream_t Sam3TextEncoderTest::stream_;

TEST_F(Sam3TextEncoderTest, ConstructsWithoutThrowing) {
    // encoder_ is already constructed in SetUpTestSuite; if that threw, the
    // whole suite would have failed at setup. This just documents the
    // expectation explicitly and re-checks the object is usable.
    ASSERT_NE(encoder_, nullptr);
}

TEST_F(Sam3TextEncoderTest, ConstructorThrowsOnBadPlanPath) {
    EXPECT_THROW(Sam3TextEncoder("/nonexistent/path/does_not_exist.plan"),
                 std::exception);
}

TEST_F(Sam3TextEncoderTest, EncodeRunsWithoutThrowing) {
    EXPECT_NO_THROW({
        Sam3TextFeatures features = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
        cudaStreamSynchronize(stream_);
        (void)features;
    });
}

TEST_F(Sam3TextEncoderTest, EncodeReturnsNonNullPointers) {
    Sam3TextFeatures features = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_NE(features.text_features, nullptr);
    EXPECT_NE(features.text_embeddings, nullptr);
}

TEST_F(Sam3TextEncoderTest, OutputPointersAreDistinct) {
    Sam3TextFeatures features = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_NE(static_cast<const void*>(features.text_features),
              static_cast<const void*>(features.text_embeddings));
}

// Regression test: buffers are allocated once in discoverAndAllocate() and
// reused across calls -- encode() should return the same device addresses
// every time, not reallocate.
TEST_F(Sam3TextEncoderTest, PointersAreStableAcrossRepeatedCalls) {
    Sam3TextFeatures first = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
    cudaStreamSynchronize(stream_);

    Sam3TextFeatures second = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(first.text_features, second.text_features);
    EXPECT_EQ(first.text_embeddings, second.text_embeddings);
}

TEST_F(Sam3TextEncoderTest, EncodeProducesFiniteOutput) {
    Sam3TextFeatures features = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    for (float v : sampleHost(features.text_features)) {
        EXPECT_TRUE(std::isfinite(v)) << "text_features produced non-finite value";
    }
    for (float v : sampleHost(features.text_embeddings)) {
        EXPECT_TRUE(std::isfinite(v)) << "text_embeddings produced non-finite value";
    }
}

// text_features: 32x1x256, text_embeddings: 32x1x1024, per the engine's
// output0/output2 bindings. Buffers are copied to host at exactly the
// expected element count; a cudaMemcpy past the end of an undersized
// allocation fails, which is what actually catches a shape mismatch here.
TEST_F(Sam3TextEncoderTest, OutputsHaveExpectedShape) {
    Sam3TextFeatures features = encoder_->encode(d_input_ids_, d_attention_mask_, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<__half> text_features_host(kTextFeaturesCount);
    ASSERT_EQ(cudaMemcpy(text_features_host.data(), features.text_features,
                          kTextFeaturesCount * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<__half> text_embeddings_host(kTextEmbeddingsCount);
    ASSERT_EQ(cudaMemcpy(text_embeddings_host.data(), features.text_embeddings,
                          kTextEmbeddingsCount * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);
}
