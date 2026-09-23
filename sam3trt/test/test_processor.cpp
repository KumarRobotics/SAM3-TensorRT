#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sam3trt/Sam3Processor.hpp"

static const std::string CONFIG_DIR = SAM3_CONFIG_DIR;

static constexpr int kImageSize = 644;
static constexpr int kTextLen = 32;

class Sam3ProcessorTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        processor_ = std::make_unique<Sam3Processor>(CONFIG_DIR + "/merges.txt", CONFIG_DIR + "/vocab.json");
        tokenizer_ = CLIPTokenizer(CONFIG_DIR + "/merges.txt", CONFIG_DIR + "/vocab.json");
    }

    static void TearDownTestSuite() {
        processor_.reset();
    }

    static cv::Mat randomImage(int rows, int cols) {
        cv::Mat image(rows, cols, CV_8UC3);
        cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(256));
        return image;
    }

    static std::vector<float> downloadImage(const Sam3ImageInput& input) {
        std::vector<float> host(3 * kImageSize * kImageSize);
        cudaMemcpy(host.data(), input.d_image, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return host;
    }

    static std::unique_ptr<Sam3Processor> processor_;
    static CLIPTokenizer tokenizer_;
};

std::unique_ptr<Sam3Processor> Sam3ProcessorTest::processor_;
CLIPTokenizer Sam3ProcessorTest::tokenizer_;

TEST_F(Sam3ProcessorTest, PreprocessImageThrowsOnEmpty) {
    EXPECT_THROW(processor_->preprocessImage(cv::Mat()), std::runtime_error);
}

TEST_F(Sam3ProcessorTest, PreprocessImageThrowsOnWrongType) {
    EXPECT_THROW(processor_->preprocessImage(cv::Mat(10, 10, CV_8UC1, cv::Scalar(0))), std::runtime_error);
    EXPECT_THROW(processor_->preprocessImage(cv::Mat(10, 10, CV_32FC3, cv::Scalar::all(0))), std::runtime_error);
}

TEST_F(Sam3ProcessorTest, PreprocessImageRecordsOriginalSize) {
    Sam3ImageInput input = processor_->preprocessImage(randomImage(480, 640));

    int32_t sizes[2] = {};
    ASSERT_EQ(cudaMemcpy(sizes, input.d_original_sizes, sizeof(sizes), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(sizes[0], 480);
    EXPECT_EQ(sizes[1], 640);
}

TEST_F(Sam3ProcessorTest, PreprocessImageMatchesCpuReference) {
    const Sam3Processor::Params params;
    const cv::Mat image = randomImage(480, 640);
    const std::vector<float> result = downloadImage(processor_->preprocessImage(image));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(kImageSize, kImageSize), 0, 0, cv::INTER_LINEAR);

    const int plane = kImageSize * kImageSize;
    for (int y = 0; y < kImageSize; ++y) {
        for (int x = 0; x < kImageSize; ++x) {
            const cv::Vec3b bgr = resized.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                const float expected = (bgr[2 - c] / 255.0f - params.image_mean[c]) / params.image_std[c];
                ASSERT_NEAR(result[c * plane + y * kImageSize + x], expected, 1e-5f)
                    << "channel " << c << " at (" << y << ", " << x << ")";
            }
        }
    }
}

TEST_F(Sam3ProcessorTest, PreprocessImagePointerIsStable) {
    const float* first = processor_->preprocessImage(randomImage(100, 100)).d_image;
    const float* second = processor_->preprocessImage(randomImage(200, 300)).d_image;
    EXPECT_EQ(first, second);
}

TEST_F(Sam3ProcessorTest, PreprocessTextThrowsOnEmpty) {
    EXPECT_THROW(processor_->preprocessText(""), std::runtime_error);
}

TEST_F(Sam3ProcessorTest, PreprocessTextPadsIdsAndMasks) {
    const std::string text = "red car";
    const std::vector<int> tokens = tokenizer_.tokenize(text);
    Sam3TextInput input = processor_->preprocessText(text);

    std::vector<int64_t> ids(kTextLen);
    std::vector<uint8_t> mask(kTextLen);
    std::vector<__half> mask_f(kTextLen);
    ASSERT_EQ(cudaMemcpy(ids.data(), input.d_input_ids, kTextLen * sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(mask.data(), input.d_attention_mask, kTextLen, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(mask_f.data(), input.d_attention_mask_f, kTextLen * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < kTextLen; ++i) {
        const bool real = i < static_cast<int>(tokens.size());
        EXPECT_EQ(ids[i], real ? tokens[i] : 0) << "id " << i;
        EXPECT_EQ(mask[i], real ? 0 : 1) << "mask " << i;
        EXPECT_EQ(__half2float(mask_f[i]), real ? 0.0f : 1.0f) << "mask_f " << i;
    }
}

TEST_F(Sam3ProcessorTest, PreprocessTextTruncatesLongPrompts) {
    std::string text;
    for (int i = 0; i < 40; ++i) text += "car ";
    Sam3TextInput input = processor_->preprocessText(text);

    std::vector<int64_t> ids(kTextLen);
    std::vector<uint8_t> mask(kTextLen);
    ASSERT_EQ(cudaMemcpy(ids.data(), input.d_input_ids, kTextLen * sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(mask.data(), input.d_attention_mask, kTextLen, cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(ids[0], 49406);
    for (int i = 0; i < kTextLen; ++i) {
        EXPECT_EQ(mask[i], 0) << "mask " << i;
    }
}

TEST_F(Sam3ProcessorTest, PreprocessReturnsOneTextInputPerPrompt) {
    Sam3Input input = processor_->preprocess(randomImage(64, 64), {"car", "person", "tree"});
    EXPECT_NE(input.image.d_image, nullptr);
    EXPECT_EQ(input.text.size(), 3u);
}
