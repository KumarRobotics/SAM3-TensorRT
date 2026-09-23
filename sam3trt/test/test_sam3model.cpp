#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "sam3trt/Sam3Model.hpp"

// Baked in at compile time by CMakeLists — see test/CMakeLists.txt
static const std::string TEST_IMAGE = TEST_IMAGE_PATH;
static const std::string CONFIG_DIR = SAM3_CONFIG_DIR;

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

// SetUpTestSuite loads the model ONCE for the whole suite.
// Loading three TRT engines per-test would add minutes of overhead.

class Sam3ModelTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string models = modelsDir();
        model_ = std::make_unique<Sam3Model>(
            models,
            CONFIG_DIR + "/merges.txt",
            CONFIG_DIR + "/vocab.json",
            CONFIG_DIR + "/params.json");

        image_ = cv::imread(TEST_IMAGE);
        ASSERT_FALSE(image_.empty())
            << "Could not load test image: " << TEST_IMAGE;
    }

    static void TearDownTestSuite() {
        model_.reset();
    }

    static std::unique_ptr<Sam3Model> model_;
    static cv::Mat                    image_;
};

std::unique_ptr<Sam3Model> Sam3ModelTest::model_;
cv::Mat Sam3ModelTest::image_;

TEST_F(Sam3ModelTest, SingleStringOverload) {
    // Convenience overload should compile and run without throwing
    EXPECT_NO_THROW({
        Sam3Result result = model_->infer(image_, "car");
        (void)result;
    });
}

TEST_F(Sam3ModelTest, VectorStringOverload) {
    EXPECT_NO_THROW({
        Sam3Result result = model_->infer(image_, {"car", "building", "road"});
        (void)result;
    });
}

TEST_F(Sam3ModelTest, FeaturesHaveCorrectPatchSize) {
    Sam3Result result = model_->infer(image_, "car");

    // patch(i, j) must return exactly 256 floats
    const auto patch = result.features.patch(0, 0);
    EXPECT_EQ(static_cast<int>(patch.size()), 256);
}

TEST_F(Sam3ModelTest, FeaturesPatchBoundsCheck) {
    Sam3Result result = model_->infer(image_, "car");

    // In-bounds should not throw
    EXPECT_NO_THROW(result.features.patch(0,  0));
    EXPECT_NO_THROW(result.features.patch(45, 45));

    // Out-of-bounds should throw
    EXPECT_THROW(result.features.patch(-1, 0),  std::out_of_range);
    EXPECT_THROW(result.features.patch(0,  -1), std::out_of_range);
    EXPECT_THROW(result.features.patch(46, 0),  std::out_of_range);
    EXPECT_THROW(result.features.patch(0,  46), std::out_of_range);
}

TEST_F(Sam3ModelTest, DetectionsSubscriptable) {
    Sam3Result result = model_->infer(image_, {"car", "building", "road"});

    // operator[] on a valid index should not throw
    if (!result.detections.empty()) {
        EXPECT_NO_THROW(result.detections[0]);
    }

    // Out-of-bounds should throw
    EXPECT_THROW(result.detections[static_cast<int>(result.detections.size())],
                 std::out_of_range);
}

TEST_F(Sam3ModelTest, DetectionsIterable) {
    Sam3Result result = model_->infer(image_, {"car", "building", "road"});

    // Range-based for loop must compile and run
    int count = 0;
    for (const Detection& det : result.detections) {
        (void)det;
        ++count;
    }
    EXPECT_EQ(count, static_cast<int>(result.detections.size()));
}

TEST_F(Sam3ModelTest, DetectionFieldsAreValid) {
    Sam3Result result = model_->infer(image_, {"car", "building", "road"});

    for (const Detection& det : result.detections) {
        // bbox must be [x, y, w, h]
        EXPECT_EQ(static_cast<int>(det.bbox.size()), 4);
        EXPECT_GE(det.bbox[2], 0) << "bbox width must be >= 0";
        EXPECT_GE(det.bbox[3], 0) << "bbox height must be >= 0";

        // mask must be CV_8UC1 at original image resolution
        EXPECT_EQ(det.mask.type(), CV_8UC1);
        EXPECT_EQ(det.mask.rows, image_.rows);
        EXPECT_EQ(det.mask.cols, image_.cols);

        // confidence must be finite and positive
        EXPECT_TRUE(std::isfinite(det.confidence));
        EXPECT_GT(det.confidence, 0.0f);

        // class_id must be one of {0, 1, 2} (car, building, road)
        EXPECT_GE(det.class_id, 0);
        EXPECT_LE(det.class_id, 2);

        // class_label must be non-empty
        EXPECT_FALSE(det.class_label.empty());
    }
}

TEST_F(Sam3ModelTest, InferThrowsOnEmptyTexts) {
    EXPECT_THROW(model_->infer(image_, std::vector<std::string>{}), std::runtime_error);
}

TEST_F(Sam3ModelTest, InferThrowsOnEmptyImage) {
    EXPECT_THROW(model_->infer(cv::Mat(), "car"), std::runtime_error);
}

TEST_F(Sam3ModelTest, DetectsCarsInTestImage) {
    Sam3Result result = model_->infer(image_, "car");

    ASSERT_FALSE(result.detections.empty());
    for (const Detection& det : result.detections) {
        EXPECT_EQ(det.class_label, "car");
        EXPECT_EQ(det.class_id, 0);
        EXPECT_GE(det.confidence, 0.3f);
        EXPECT_LE(det.confidence, 1.0f);
    }
}

TEST_F(Sam3ModelTest, ClassIdsMatchPromptOrder) {
    const std::vector<std::string> texts = {"person", "car"};
    Sam3Result result = model_->infer(image_, texts);

    std::set<int> seen;
    for (const Detection& det : result.detections) {
        ASSERT_GE(det.class_id, 0);
        ASSERT_LT(det.class_id, static_cast<int>(texts.size()));
        EXPECT_EQ(det.class_label, texts[det.class_id]);
        seen.insert(det.class_id);
    }
    EXPECT_EQ(seen, (std::set<int>{0, 1}));
}

TEST_F(Sam3ModelTest, BboxMatchesMaskBoundingRect) {
    Sam3Result result = model_->infer(image_, "car");

    for (const Detection& det : result.detections) {
        const cv::Rect rect = cv::boundingRect(det.mask);
        EXPECT_EQ(det.bbox, (std::vector<int>{rect.x, rect.y, rect.width, rect.height}));
        EXPECT_GT(cv::countNonZero(det.mask), 0);
    }
}

TEST_F(Sam3ModelTest, MasksAreBinary) {
    Sam3Result result = model_->infer(image_, "car");

    for (const Detection& det : result.detections) {
        const int on = cv::countNonZero(det.mask == 255);
        const int off = cv::countNonZero(det.mask == 0);
        EXPECT_EQ(on + off, det.mask.rows * det.mask.cols);
    }
}

TEST_F(Sam3ModelTest, FeaturesAreFiniteAndBounded) {
    Sam3Result result = model_->infer(image_, "car");
    const std::vector<float>& data = result.features.data();

    float min_v = data[0], max_v = data[0];
    for (float v : data) {
        ASSERT_TRUE(std::isfinite(v));
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    EXPECT_LT(std::max(std::abs(min_v), std::abs(max_v)), 1000.0f);
    EXPECT_GT(max_v - min_v, 1e-3f);
}

TEST_F(Sam3ModelTest, InferIsDeterministic) {
    Sam3Result first = model_->infer(image_, "car");
    Sam3Result second = model_->infer(image_, "car");

    ASSERT_EQ(first.detections.size(), second.detections.size());
    for (size_t i = 0; i < first.detections.size(); ++i) {
        EXPECT_EQ(first.detections[i].bbox, second.detections[i].bbox);
        EXPECT_FLOAT_EQ(first.detections[i].confidence, second.detections[i].confidence);
    }
    EXPECT_EQ(first.features.data(), second.features.data());
}
