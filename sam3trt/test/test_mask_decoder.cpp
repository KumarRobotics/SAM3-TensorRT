#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sam3trt/Sam3MaskDecoder.hpp"

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

static std::string maskDecoderPlanPath() {
    return modelsDir() + "/mask_decoder_fp16.engine";
}

// Input shapes, per the engine's fpn/pos/txt bindings.
static constexpr size_t kFpnCount[3] = {
    256 * 184 * 184,  // fpn0 / pos0
    256 * 92 * 92,    // fpn1 / pos1
    256 * 46 * 46,    // fpn2 / pos2
};
static constexpr size_t kTxtFeatsCount = 32 * 1 * 256;
static constexpr size_t kTxtLen = 32;  // txt_masks / txt_masks_f

// Output shapes, per the engine's output0-4 bindings.
static constexpr size_t kPredictedLogitsCount = 1 * 200 * 1;        // output0
static constexpr size_t kPredictedBoxesCount = 1 * 200 * 4;         // output1
static constexpr size_t kPredictedBoxesXyxyCount = 1 * 200 * 4;     // output2
static constexpr size_t kPresenceLogitsCount = 1 * 1;               // output3
static constexpr size_t kPredictedMasksCount = 1 * 200 * 184 * 184; // output4

static constexpr int kSampleCount = 16;

static __half* mallocZeroedHalf(size_t count) {
    __half* d_ptr = nullptr;
    if (cudaMalloc(&d_ptr, count * sizeof(__half)) != cudaSuccess) {
        throw std::runtime_error("mallocZeroedHalf: cudaMalloc failed");
    }
    // fp16 zero is the 2-byte pattern 0x0000, same as a plain cudaMemset(0).
    if (cudaMemset(d_ptr, 0, count * sizeof(__half)) != cudaSuccess) {
        throw std::runtime_error("mallocZeroedHalf: cudaMemset failed");
    }
    return d_ptr;
}

class Sam3MaskDecoderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        decoder_ = std::make_unique<Sam3MaskDecoder>(maskDecoderPlanPath());

        for (int i = 0; i < 3; ++i) {
            image_features_.fpn[i] = mallocZeroedHalf(kFpnCount[i]);
            image_features_.fpn_pos[i] = mallocZeroedHalf(kFpnCount[i]);
        }

        text_features_.text_features = nullptr;  // unused by decode()
        text_features_.text_embeddings = mallocZeroedHalf(kTxtFeatsCount);

        ASSERT_EQ(cudaMalloc(&d_attention_mask_, kTxtLen * sizeof(bool)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_attention_mask_, 0, kTxtLen * sizeof(bool)), cudaSuccess);

        d_attention_mask_f_ = mallocZeroedHalf(kTxtLen);

        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    static void TearDownTestSuite() {
        cudaStreamDestroy(stream_);

        for (int i = 0; i < 3; ++i) {
            cudaFree(image_features_.fpn[i]);
            cudaFree(image_features_.fpn_pos[i]);
        }
        cudaFree(text_features_.text_embeddings);
        cudaFree(d_attention_mask_);
        cudaFree(d_attention_mask_f_);

        decoder_.reset();
    }

    static std::vector<float> sampleHost(const float* d_ptr, size_t n = kSampleCount) {
        std::vector<float> host(n);
        cudaMemcpy(host.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
        return host;
    }

    static std::unique_ptr<Sam3MaskDecoder> decoder_;
    static Sam3ImageFeatures                image_features_;
    static Sam3TextFeatures                 text_features_;
    static bool*                            d_attention_mask_;
    static __half*                          d_attention_mask_f_;
    static cudaStream_t                     stream_;
};

std::unique_ptr<Sam3MaskDecoder> Sam3MaskDecoderTest::decoder_;
Sam3ImageFeatures Sam3MaskDecoderTest::image_features_{};
Sam3TextFeatures Sam3MaskDecoderTest::text_features_{};
bool* Sam3MaskDecoderTest::d_attention_mask_ = nullptr;
__half* Sam3MaskDecoderTest::d_attention_mask_f_ = nullptr;
cudaStream_t Sam3MaskDecoderTest::stream_;

TEST_F(Sam3MaskDecoderTest, ConstructsWithoutThrowing) {
    ASSERT_NE(decoder_, nullptr);
}

TEST_F(Sam3MaskDecoderTest, ConstructorThrowsOnBadPlanPath) {
    EXPECT_THROW(Sam3MaskDecoder("/nonexistent/path/does_not_exist.plan"),
                 std::exception);
}

TEST_F(Sam3MaskDecoderTest, DecodeRunsWithoutThrowing) {
    EXPECT_NO_THROW({
        Sam3DecoderOutput<float> out = decoder_->decode(
            image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
        cudaStreamSynchronize(stream_);
        (void)out;
    });
}

TEST_F(Sam3MaskDecoderTest, DecodeReturnsNonNullPointers) {
    Sam3DecoderOutput<float> out = decoder_->decode(
        image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_NE(out.predicted_logits, nullptr);
    EXPECT_NE(out.predicted_boxes, nullptr);
    EXPECT_NE(out.predicted_boxes_xyxy, nullptr);
    EXPECT_NE(out.presence_logits, nullptr);
    EXPECT_NE(out.predicted_masks, nullptr);
}

TEST_F(Sam3MaskDecoderTest, OutputPointersAreAllDistinct) {
    Sam3DecoderOutput<float> out = decoder_->decode(
        image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<const void*> ptrs = {
        out.predicted_logits, out.predicted_boxes, out.predicted_boxes_xyxy,
        out.presence_logits, out.predicted_masks,
    };

    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]) << "buffer " << i << " and buffer " << j << " alias";
        }
    }
}

// Regression test: output buffers are allocated once in discoverAndAllocate()
// and reused across calls -- decode() should return the same device
// addresses every time, not reallocate.
TEST_F(Sam3MaskDecoderTest, PointersAreStableAcrossRepeatedCalls) {
    Sam3DecoderOutput<float> first = decoder_->decode(
        image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
    cudaStreamSynchronize(stream_);

    Sam3DecoderOutput<float> second = decoder_->decode(
        image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(first.predicted_logits, second.predicted_logits);
    EXPECT_EQ(first.predicted_boxes, second.predicted_boxes);
    EXPECT_EQ(first.predicted_boxes_xyxy, second.predicted_boxes_xyxy);
    EXPECT_EQ(first.presence_logits, second.presence_logits);
    EXPECT_EQ(first.predicted_masks, second.predicted_masks);
}

TEST_F(Sam3MaskDecoderTest, DecodeProducesFiniteOutput) {
    Sam3DecoderOutput<float> out = decoder_->decode(
        image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    for (float v : sampleHost(out.predicted_logits, std::min<size_t>(kSampleCount, kPredictedLogitsCount))) {
        EXPECT_TRUE(std::isfinite(v)) << "predicted_logits produced non-finite value";
    }
    for (float v : sampleHost(out.predicted_boxes, std::min<size_t>(kSampleCount, kPredictedBoxesCount))) {
        EXPECT_TRUE(std::isfinite(v)) << "predicted_boxes produced non-finite value";
    }
    for (float v : sampleHost(out.predicted_boxes_xyxy, std::min<size_t>(kSampleCount, kPredictedBoxesXyxyCount))) {
        EXPECT_TRUE(std::isfinite(v)) << "predicted_boxes_xyxy produced non-finite value";
    }
    for (float v : sampleHost(out.presence_logits, std::min<size_t>(kSampleCount, kPresenceLogitsCount))) {
        EXPECT_TRUE(std::isfinite(v)) << "presence_logits produced non-finite value";
    }
    for (float v : sampleHost(out.predicted_masks, std::min<size_t>(kSampleCount, kPredictedMasksCount))) {
        EXPECT_TRUE(std::isfinite(v)) << "predicted_masks produced non-finite value";
    }
}

// Shapes per output0-4 bindings: [1,200,1], [1,200,4], [1,200,4], [1,1],
// [1,200,184,184]. Buffers are copied to host at exactly the expected
// element count; a cudaMemcpy past the end of an undersized allocation
// fails, which is what actually catches a shape mismatch here.
TEST_F(Sam3MaskDecoderTest, OutputsHaveExpectedShape) {
    Sam3DecoderOutput<float> out = decoder_->decode(
        image_features_, text_features_, d_attention_mask_, d_attention_mask_f_, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float> logits_host(kPredictedLogitsCount);
    ASSERT_EQ(cudaMemcpy(logits_host.data(), out.predicted_logits,
                          kPredictedLogitsCount * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<float> boxes_host(kPredictedBoxesCount);
    ASSERT_EQ(cudaMemcpy(boxes_host.data(), out.predicted_boxes,
                          kPredictedBoxesCount * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<float> boxes_xyxy_host(kPredictedBoxesXyxyCount);
    ASSERT_EQ(cudaMemcpy(boxes_xyxy_host.data(), out.predicted_boxes_xyxy,
                          kPredictedBoxesXyxyCount * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<float> presence_host(kPresenceLogitsCount);
    ASSERT_EQ(cudaMemcpy(presence_host.data(), out.presence_logits,
                          kPresenceLogitsCount * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<float> masks_host(kPredictedMasksCount);
    ASSERT_EQ(cudaMemcpy(masks_host.data(), out.predicted_masks,
                          kPredictedMasksCount * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
}
