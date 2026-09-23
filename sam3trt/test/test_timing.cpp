#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sam3trt/Sam3Model.hpp"

static const std::string TEST_IMAGE = TEST_IMAGE_PATH;
static const std::string CONFIG_DIR = SAM3_CONFIG_DIR;

static std::string modelsDir() {
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("HOME not set");
    return std::string(home) + "/models";
}

class Sam3TimingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        model_ = std::make_unique<Sam3Model>(
            modelsDir(),
            CONFIG_DIR + "/merges.txt",
            CONFIG_DIR + "/vocab.json",
            CONFIG_DIR + "/params.json");

        image_ = cv::imread(TEST_IMAGE);
        ASSERT_FALSE(image_.empty())
            << "Could not load test image: " << TEST_IMAGE;

        // All 10 prompts — subsets are taken as first N elements
        all_texts_ = {
            "car", "building", "road", "person", "tree",
            "sky", "window",   "door", "sign",   "sidewalk"
        };
    }

    static void TearDownTestSuite() { model_.reset(); }

    using Clock    = std::chrono::high_resolution_clock;
    using Ms       = std::chrono::duration<double, std::milli>;

    static std::unique_ptr<Sam3Model>  model_;
    static cv::Mat                     image_;
    static std::vector<std::string>    all_texts_;

    static constexpr int WARMUP_RUNS  = 3;
    static constexpr int MEASURE_RUNS = 10;
};

std::unique_ptr<Sam3Model> Sam3TimingTest::model_;
cv::Mat                    Sam3TimingTest::image_;
std::vector<std::string>   Sam3TimingTest::all_texts_;

TEST_F(Sam3TimingTest, ForwardPassLatency) {
    std::cout << "\n"
              << std::string(52, '-') << "\n"
              << std::left
              << std::setw(14) << "  Texts"
              << std::setw(14) << "  Min (ms)"
              << std::setw(14) << "  Max (ms)"
              << std::setw(14) << "  Avg (ms)"  << "\n"
              << std::string(52, '-') << "\n";

    for (int n = 1; n <= 10; ++n) {
        const std::vector<std::string> texts(
            all_texts_.begin(), all_texts_.begin() + n);

        // Warmup — avoids measuring CUDA cold-start on first call
        for (int i = 0; i < WARMUP_RUNS; ++i)
            model_->infer(image_, texts);

        // Measured runs
        double total_ms = 0.0;
        double min_ms   = std::numeric_limits<double>::max();
        double max_ms   = 0.0;

        for (int i = 0; i < MEASURE_RUNS; ++i) {
            const auto t0  = Clock::now();
            model_->infer(image_, texts);
            const double ms = Ms(Clock::now() - t0).count();

            total_ms += ms;
            min_ms    = std::min(min_ms, ms);
            max_ms    = std::max(max_ms, ms);
        }

        const double avg_ms = total_ms / MEASURE_RUNS;

        std::cout << "  "
                  << std::left  << std::setw(12) << n
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << min_ms
                  << std::setw(14) << max_ms
                  << std::setw(14) << avg_ms << "\n";

        // Loose sanity check — 30 seconds per call would indicate something
        // is seriously wrong rather than just slow hardware
        EXPECT_LT(avg_ms, 30000.0)
            << n << " text(s): average latency exceeded 30s";
    }

    std::cout << std::string(52, '-') << "\n\n";
}
