#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <vector>

#include "sam3trt/cuda/normalize.cuh"
#include "sam3trt/cuda/preprocess.cuh"
#include "sam3trt/cuda/typing.cuh"

static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3] = {0.229f, 0.224f, 0.225f};

static cv::Mat randomImage(int rows, int cols) {
    cv::Mat image(rows, cols, CV_8UC3);
    cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(256));
    return image;
}

static std::vector<float> runRgbKernel(const cv::Mat& image, int dst_h, int dst_w, bool fused) {
    const size_t src_bytes = image.total() * 3;
    const size_t dst_count = static_cast<size_t>(3) * dst_h * dst_w;

    uint8_t* d_src = nullptr;
    float* d_dst = nullptr;
    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_dst, dst_count * sizeof(float));
    cudaMemcpy(d_src, image.data, src_bytes, cudaMemcpyHostToDevice);

    if (fused) {
        preprocessFused(d_src, d_dst, image.rows, image.cols, dst_h, dst_w, kMean, kStd);
    } else {
        normalizeImage(d_src, d_dst, dst_h, dst_w, kMean, kStd, nullptr);
    }

    std::vector<float> host(dst_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(host.data(), d_dst, dst_count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    return host;
}

static void expectMatchesReference(const std::vector<float>& result, const cv::Mat& bgr_float, float tolerance) {
    const int h = bgr_float.rows;
    const int w = bgr_float.cols;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f bgr = bgr_float.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                const float expected = (bgr[2 - c] / 255.0f - kMean[c]) / kStd[c];
                ASSERT_NEAR(result[(c * h + y) * w + x], expected, tolerance)
                    << "channel " << c << " at (" << y << ", " << x << ")";
            }
        }
    }
}

TEST(NormalizeImageTest, MatchesCpuReference) {
    const cv::Mat image = randomImage(37, 53);
    cv::Mat image_f;
    image.convertTo(image_f, CV_32FC3);

    expectMatchesReference(runRgbKernel(image, image.rows, image.cols, false), image_f, 1e-5f);
}

TEST(PreprocessFusedTest, DownscaleMatchesOpenCv) {
    const cv::Mat image = randomImage(480, 640);
    cv::Mat image_f, resized;
    image.convertTo(image_f, CV_32FC3);
    cv::resize(image_f, resized, cv::Size(644, 644), 0, 0, cv::INTER_LINEAR);

    expectMatchesReference(runRgbKernel(image, 644, 644, true), resized, 1e-3f);
}

TEST(PreprocessFusedTest, UpscaleMatchesOpenCv) {
    const cv::Mat image = randomImage(50, 70);
    cv::Mat image_f, resized;
    image.convertTo(image_f, CV_32FC3);
    cv::resize(image_f, resized, cv::Size(211, 157), 0, 0, cv::INTER_LINEAR);

    expectMatchesReference(runRgbKernel(image, 157, 211, true), resized, 1e-3f);
}

TEST(PreprocessFusedTest, IdentitySizeMatchesNormalize) {
    const cv::Mat image = randomImage(64, 96);
    EXPECT_EQ(runRgbKernel(image, 64, 96, true), runRgbKernel(image, 64, 96, false));
}

class TypingTest : public ::testing::Test {
protected:
    void SetUp() override {
        values_.resize(kCount);
        for (size_t i = 0; i < kCount; ++i) {
            values_[i] = std::sin(static_cast<float>(i)) * 100.0f;
        }
        cudaMalloc(&d_in_, kCount * sizeof(float));
        cudaMalloc(&d_native_, kCount * sizeof(float));
        cudaMalloc(&d_out_, kCount * sizeof(float));
        cudaMemcpy(d_in_, values_.data(), kCount * sizeof(float), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_in_);
        cudaFree(d_native_);
        cudaFree(d_out_);
    }

    std::vector<float> roundTrip(nvinfer1::DataType dt) {
        floatToNative(d_in_, d_native_, kCount, dt, nullptr);
        nativeToFloat(d_native_, d_out_, kCount, dt, nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::vector<float> out(kCount);
        cudaMemcpy(out.data(), d_out_, kCount * sizeof(float), cudaMemcpyDeviceToHost);
        return out;
    }

    static constexpr size_t kCount = 1000;
    std::vector<float> values_;
    float* d_in_ = nullptr;
    void* d_native_ = nullptr;
    float* d_out_ = nullptr;
};

TEST_F(TypingTest, FloatRoundTripIsExact) {
    EXPECT_EQ(roundTrip(nvinfer1::DataType::kFLOAT), values_);
}

TEST_F(TypingTest, HalfRoundTripMatchesHostConversion) {
    const std::vector<float> out = roundTrip(nvinfer1::DataType::kHALF);
    for (size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(out[i], __half2float(__float2half(values_[i]))) << "index " << i;
    }
}

TEST_F(TypingTest, UnsupportedTypeThrows) {
    EXPECT_THROW(floatToNative(d_in_, d_native_, kCount, nvinfer1::DataType::kINT32, nullptr), std::runtime_error);
    EXPECT_THROW(nativeToFloat(d_native_, d_out_, kCount, nvinfer1::DataType::kINT32, nullptr), std::runtime_error);
}
