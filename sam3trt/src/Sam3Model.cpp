#include "sam3trt/Sam3Model.hpp"
#include "sam3trt/cuda/upsample.cuh"

#include <json/json.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <fstream>

namespace {
// predicted_logits / presence_logits are raw, pre-activation logits from the
// model -- unbounded, not probabilities. They must be squashed through a
// sigmoid before comparing against a [0,1]-scaled threshold from config.
inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
}  // namespace


Sam3Model::Config Sam3Model::loadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Sam3Model: cannot open config: " + path);
    }

    Json::Value  root;
    Json::Reader reader;
    if (!reader.parse(file, root)) {
        throw std::runtime_error("Sam3Model: failed to parse config: " + reader.getFormattedErrorMessages());
    }
 
    Config cfg;
    cfg.presence_threshold = root["presence_threshold"].asFloat();
    cfg.confidence_threshold = root["confidence_threshold"].asFloat();
    cfg.image_size = root["image_size"].asInt();
    cfg.text_padding = root["text_padding"].asInt();
    cfg.max_img_h = root["max_img_h"].asInt();
    cfg.max_img_w = root["max_img_w"].asInt();
 
    const Json::Value mean = root["image_mean"];
    const Json::Value std  = root["image_std"];
    for (int i = 0; i < 3; ++i) {
        cfg.image_mean[i] = mean[i].asFloat();
        cfg.image_std[i]  = std[i].asFloat();
    }
 
    return cfg;
}

Sam3Processor::Params Sam3Model::processorParams(const Config& cfg) {
    Sam3Processor::Params p;
    p.image_size   = cfg.image_size;
    p.text_padding = cfg.text_padding;

    for (int i = 0; i < 3; ++i) {
        p.image_mean[i] = cfg.image_mean[i];
        p.image_std[i]  = cfg.image_std[i];
    }
    
    return p;
}

Sam3Model::Sam3Model(const std::string& models_path, const std::string& merges_path, const std::string& vocab_path, const std::string& config_path) :
    config_(loadConfig(config_path)),
    image_encoder_(models_path + "/image_encoder_fp16.engine"),
    text_encoder_(models_path + "/text_encoder_fp16.engine"),
    mask_decoder_(models_path + "/mask_decoder_fp16.engine"),
    processor_(merges_path, vocab_path, processorParams(config_)),
    h_logits_(MAX_SLOTS),
    h_presence_(1)
{
    cudaStreamCreate(&img_stream_);
    cudaStreamCreate(&inference_stream_);
    allocateBuffers();
}

Sam3Model::~Sam3Model() {
    cudaStreamDestroy(img_stream_);
    cudaStreamDestroy(inference_stream_);
    cudaFree(d_indices_);
    cudaFree(d_upsampled_);
    cudaFree(d_features_);
}

void Sam3Model::allocateBuffers() {
    cudaMalloc(&d_indices_, MAX_SLOTS * sizeof(int));
    cudaMalloc(&d_upsampled_, static_cast<size_t>(MAX_SLOTS) * config_.max_img_h * config_.max_img_w * sizeof(float));
    cudaMalloc(&d_features_, static_cast<size_t>(FeatureMap::C) * FeatureMap::H * FeatureMap::W * sizeof(float));
}

std::vector<Detection> Sam3Model::forward(
    const Sam3ImageFeatures& image_features,
    const Sam3ImageInput& image_input,
    const std::string& text,
    int class_id,
    int orig_h,
    int orig_w)
{
    Sam3TextInput text_input = processor_.preprocessText(text);

    Sam3TextFeatures text_features = text_encoder_.encode(
        text_input.d_input_ids,
        text_input.d_attention_mask,
        inference_stream_);

    Sam3DecoderOutput<float> output = mask_decoder_.decode(
        image_features,
        text_features,
        text_input.d_attention_mask,
        text_input.d_attention_mask_f,
        inference_stream_);

    {
        cudaError_t err = cudaStreamSynchronize(inference_stream_);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] decode sync failed: " << cudaGetErrorString(err) << "\n";
        }
    }
    cudaMemcpy(h_logits_.data(), output.predicted_logits, MAX_SLOTS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_presence_.data(), output.presence_logits, sizeof(float), cudaMemcpyDeviceToHost);

    const float presence_prob = sigmoidf(h_presence_[0]);
    if (presence_prob < config_.presence_threshold) {
        return {};
    }

    std::vector<int>   surviving_indices;
    std::vector<float> surviving_confidences;
    for (int i = 0; i < MAX_SLOTS; ++i) {
        const float confidence_prob = sigmoidf(h_logits_[i]);
        if (confidence_prob >= config_.confidence_threshold) {
            surviving_indices.push_back(i);
            surviving_confidences.push_back(confidence_prob);
        }
    }

    if (surviving_indices.empty()) return {};

    const int N = static_cast<int>(surviving_indices.size());

    cudaMemcpy(d_indices_, surviving_indices.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    upsampleMasks(output.predicted_masks, d_indices_, d_upsampled_,
                  N, SRC_MASK_H, SRC_MASK_W,
                  orig_h, orig_w, inference_stream_);

    {
        cudaError_t err = cudaStreamSynchronize(inference_stream_);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] upsampleMasks sync failed: " << cudaGetErrorString(err) << "\n";
        }
    }

    const size_t mask_pixels = static_cast<size_t>(orig_h) * orig_w;
    std::vector<float> mask_f(mask_pixels);

    std::vector<Detection> detections;
    detections.reserve(N);

    for (int i = 0; i < N; ++i) {
        cudaMemcpy(mask_f.data(),
                   d_upsampled_ + static_cast<size_t>(i) * mask_pixels,
                   mask_pixels * sizeof(float),
                   cudaMemcpyDeviceToHost);

        cv::Mat mask_float(orig_h, orig_w, CV_32FC1, mask_f.data());
        cv::Mat mask_bin;
        cv::threshold(mask_float, mask_bin, 0.0f, 255.0f, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8UC1);

        const cv::Rect bbox = cv::boundingRect(mask_bin);

        Detection det;
        det.bbox = {bbox.x, bbox.y, bbox.width, bbox.height};
        det.mask = mask_bin.clone();
        det.class_label = text;
        det.class_id = class_id;
        det.confidence  = surviving_confidences[i];

        detections.push_back(std::move(det));
    }

    return detections;
}

Sam3Result Sam3Model::infer(const cv::Mat& image, const std::string& text) {
    return infer(image, std::vector<std::string>{text});
}

Sam3Result Sam3Model::infer(const cv::Mat& image, const std::vector<std::string>& texts)
{
    if (texts.empty()) {
        throw std::runtime_error("Sam3Model::infer: texts must not be empty");
    }

    Sam3ImageInput image_input = processor_.preprocessImage(image);

    Sam3ImageFeatures image_features = image_encoder_.encode(image_input.d_image, img_stream_);
    {
        cudaError_t err = cudaStreamSynchronize(img_stream_);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] image_encoder sync failed: " << cudaGetErrorString(err) << "\n";
        }
    }

    constexpr size_t feature_size = FeatureMap::C * FeatureMap::H * FeatureMap::W;
    std::vector<float> feature_data(feature_size);
    nativeToFloat(image_features.fpn[FEATURE_IDX], d_features_, feature_size, nvinfer1::DataType::kHALF, img_stream_);
    cudaMemcpyAsync(feature_data.data(), d_features_, feature_size * sizeof(float), cudaMemcpyDeviceToHost, img_stream_);
    cudaStreamSynchronize(img_stream_);

    int32_t orig_sizes[2];
    cudaMemcpy(orig_sizes, image_input.d_original_sizes, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    const int orig_h = static_cast<int>(orig_sizes[0]);
    const int orig_w = static_cast<int>(orig_sizes[1]);

    std::vector<Detection> all_detections;
    for (int i = 0; i < static_cast<int>(texts.size()); ++i) {
        auto dets = forward(image_features, image_input, texts[i], i, orig_h, orig_w);
        for (auto& d : dets) {
            all_detections.push_back(std::move(d));
        }
    }

    return Sam3Result{
        Detections(std::move(all_detections)),
        FeatureMap(std::move(feature_data))
    };
}
