#pragma once

#include "sam3trt/Sam3ImageEncoder.hpp"
#include "sam3trt/Sam3TextEncoder.hpp"
#include "sam3trt/Sam3MaskDecoder.hpp"
#include "sam3trt/Sam3Processor.hpp"
#include "sam3trt/Sam3Result.hpp"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Sam3Model {
    public:
        /**
         * @param models_path  Directory containing image_encoder.plan,
         *                     text_encoder.plan, mask_decoder.plan
         * @param merges_path  Path to BPE merges file for CLIPTokenizer
         * @param vocab_path   Path to vocab file for CLIPTokenizer
         * @param config_path  Path to YAML config file
         */
        Sam3Model(const std::string& models_path,
                  const std::string& merges_path,
                  const std::string& vocab_path,
                  const std::string& config_path);
        ~Sam3Model();

        Sam3Model(const Sam3Model&) = delete;
        Sam3Model& operator=(const Sam3Model&) = delete;

        /** Single text prompt — convenience overload. */
        Sam3Result infer(const cv::Mat& image, const std::string& text);

        /** Multiple text prompts — encodes image once, loops over texts. */
        Sam3Result infer(const cv::Mat& image, const std::vector<std::string>& texts);

    private:
        struct Config {
            float presence_threshold   = 0.5f;
            float confidence_threshold = 0.3f;
            float image_mean[3]        = {0.485f, 0.456f, 0.406f};
            float image_std[3]         = {0.229f, 0.224f, 0.225f};
            int image_size           = 644;
            int text_padding         = 32;
            int max_img_h            = 1536;
            int max_img_w            = 2048;
        };

        static Config loadConfig(const std::string& path);
        static Sam3Processor::Params processorParams(const Config& cfg);

        void allocateBuffers();

        /**
         * Runs text encode + decode + threshold + upsample for one prompt.
         * Image features are already on GPU from the caller.
         * Returns detections for this prompt — empty if nothing survives thresholds.
         */
        std::vector<Detection> forward(const Sam3ImageFeatures& image_features,
                                       const Sam3ImageInput& image_input,
                                       const std::string& text,
                                       int class_id,
                                       int orig_h,
                                       int orig_w);

        Config config_;

        Sam3ImageEncoder image_encoder_;
        Sam3TextEncoder text_encoder_;
        Sam3MaskDecoder mask_decoder_;
        Sam3Processor processor_;

        cudaStream_t img_stream_ = nullptr;  // image encoder only
        cudaStream_t inference_stream_ = nullptr;  // text encoder + decoder loop

        // GPU buffers for upsampling allocated once for worst case
        int* d_indices_ = nullptr;  // [200]                       int32
        float* d_upsampled_ = nullptr;  // [200, max_img_h, max_img_w] float32
        float* d_features_ = nullptr;

        // Host buffers for thresholding tiny, no need for pinned memory
        std::vector<float> h_logits_; // [200]
        std::vector<float> h_presence_; // [1]

        static constexpr int MAX_SLOTS = 200;
        static constexpr int SRC_MASK_H = 184;
        static constexpr int SRC_MASK_W = 184;
        static constexpr int FEATURE_IDX = 2;    // fpn[2] = [256, 46, 46]
};
