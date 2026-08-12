#pragma once

#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <string>
#include <vector>

/**
 * Wraps the deepest FPN feature layer [256, 46, 46] in CHW layout.
 * patch(i, j) returns the 256-dim feature vector at spatial location (i, j).
 */
class FeatureMap {
    public:
        static constexpr int C = 256;
        static constexpr int H = 46;
        static constexpr int W = 46;

        explicit FeatureMap(std::vector<float> data): data_(std::move(data))
        {
            if (static_cast<int>(data_.size()) != C * H * W) {
                throw std::runtime_error(
                    "FeatureMap: expected " + std::to_string(C * H * W) +
                    " floats, got " + std::to_string(data_.size()));
            }
        }

        /**
         * Returns the 256-dim feature vector at spatial position (i, j).
         * @param i  Row    in [0, 45]
         * @param j  Column in [0, 45]
         */
        std::vector<float> patch(int i, int j) const {
            if (i < 0 || i >= H || j < 0 || j >= W) {
                throw std::out_of_range(
                    "FeatureMap::patch: (" + std::to_string(i) + ", " +
                    std::to_string(j) + ") out of range [0," +
                    std::to_string(H) + ") x [0," + std::to_string(W) + ")");
            }

            // CHW layout: data_[c * H * W + i * W + j]
            std::vector<float> vec(C);
            for (int c = 0; c < C; ++c) {
                vec[c] = data_[c * H * W + i * W + j];
            }

            return vec;  // moved by the compiler
        }

        // Raw access for serialisation or further processing
        const std::vector<float>& data() const { return data_; }

    private:
        std::vector<float> data_;
};

/**
 * A single detection from Sam3Model.
 */
struct Detection {
    std::vector<int> bbox;         // [x, y, w, h] in original image pixels
    cv::Mat mask;         // CV_8UC1 binary mask, original image resolution
    std::string class_label;  // text prompt that produced this detection
    int class_id;     // index of the text prompt
    float confidence;   // score from predicted_logits
};

/**
 * Subscriptable, iterable collection of Detection objects.
 *
 * Usage:
 *   result.detections[0].bbox
 *   for (const Detection& det : result.detections) { ... }
 */
class Detections {
    public:
        explicit Detections(std::vector<Detection> detections) : detections_(std::move(detections)) {}

        const Detection& operator[](int index) const {
            if (index < 0 || index >= static_cast<int>(detections_.size())) {
                throw std::out_of_range(
                    "Detections: index " + std::to_string(index) +
                    " out of range [0, " + std::to_string(detections_.size()) + ")");
            }
            return detections_[index];
        }

        std::size_t size() const { return detections_.size();  }
        bool empty() const { return detections_.empty(); }

        // Range-based for loop support
        std::vector<Detection>::const_iterator begin() const { return detections_.begin(); }
        std::vector<Detection>::const_iterator end() const { return detections_.end(); }

    private:
        std::vector<Detection> detections_;
};

/**
 * Top-level result returned by Sam3Processor::postprocess().
 *
 * Usage:
 *   Sam3Result result = processor.postprocess(output, text_prompts);
 *   result.detections[0].mask;
 *   result.features.patch(23, 12);
 *   for (const Detection& det : result.detections) { ... }
 */
struct Sam3Result {
    Detections detections;
    FeatureMap features;
};
