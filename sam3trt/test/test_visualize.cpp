#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <vector>

#include "sam3trt/Visualize.hpp"

static constexpr int kRows = 120;
static constexpr int kCols = 160;

static FeatureMap emptyFeatures() {
    return FeatureMap(std::vector<float>(FeatureMap::C * FeatureMap::H * FeatureMap::W, 0.0f));
}

static Detection boxDetection(int x, int y, int w, int h, int class_id) {
    Detection det;
    det.mask = cv::Mat::zeros(kRows, kCols, CV_8UC1);
    det.mask(cv::Rect(x, y, w, h)).setTo(255);
    det.bbox = {x, y, w, h};
    det.class_label = "car";
    det.class_id = class_id;
    det.confidence = 0.9f;
    return det;
}

static cv::Mat grayImage() {
    return cv::Mat(kRows, kCols, CV_8UC3, cv::Scalar(100, 100, 100));
}

TEST(VisualizeTest, ThrowsOnEmptyImage) {
    Sam3Result result{Detections({}), emptyFeatures()};
    EXPECT_THROW(visualize(cv::Mat(), result), std::invalid_argument);
}

TEST(VisualizeTest, NoDetectionsReturnsCopy) {
    const cv::Mat image = grayImage();
    Sam3Result result{Detections({}), emptyFeatures()};

    const cv::Mat output = visualize(image, result);
    EXPECT_EQ(cv::norm(output, image, cv::NORM_INF), 0.0);
    EXPECT_NE(output.data, image.data);
}

TEST(VisualizeTest, GrayscaleInputBecomesBgr) {
    const cv::Mat image(kRows, kCols, CV_8UC1, cv::Scalar(100));
    Sam3Result result{Detections({}), emptyFeatures()};

    const cv::Mat output = visualize(image, result);
    EXPECT_EQ(output.type(), CV_8UC3);
    EXPECT_EQ(output.size(), image.size());
}

TEST(VisualizeTest, DrawsInsideMaskOnly) {
    const cv::Mat image = grayImage();
    Sam3Result result{Detections({boxDetection(60, 50, 40, 40, 0)}), emptyFeatures()};

    const cv::Mat output = visualize(image, result);
    EXPECT_EQ(output.size(), image.size());
    EXPECT_NE(output.at<cv::Vec3b>(70, 80), image.at<cv::Vec3b>(70, 80));
    EXPECT_EQ(output.at<cv::Vec3b>(110, 5), image.at<cv::Vec3b>(110, 5));
}

TEST(VisualizeTest, ClassesGetDifferentColors) {
    const cv::Mat image = grayImage();
    Sam3Result result{
        Detections({boxDetection(10, 40, 40, 40, 0), boxDetection(100, 40, 40, 40, 1)}),
        emptyFeatures()};

    const cv::Mat output = visualize(image, result);
    EXPECT_NE(output.at<cv::Vec3b>(60, 30), output.at<cv::Vec3b>(60, 120));
}

TEST(VisualizeTest, ZeroAlphaLeavesMaskInteriorUnchanged) {
    const cv::Mat image = grayImage();
    Sam3Result result{Detections({boxDetection(60, 50, 40, 40, 0)}), emptyFeatures()};

    const cv::Mat output = visualize(image, result, 0.0);
    EXPECT_EQ(output.at<cv::Vec3b>(70, 80), image.at<cv::Vec3b>(70, 80));
}
