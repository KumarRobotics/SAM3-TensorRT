#pragma once

#include "sam3tensorrt/Sam3ModelBase.hpp"
#include "sam3tensorrt/Sam3ImageEncoder.hpp"
#include "sam3tensorrt/Sam3TextEncoder.hpp"

#include <string>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * Device pointers to mask decoder outputs.
 * Owned by Sam3MaskDecoder, valid until the next decode() call or destruction.
 * Caller must sync the stream before reading.*/
struct Sam3DecoderOutput {
    float* predicted_logits; 
    float* predicted_boxes;       
    float* predicted_boxes_xyxy;  
    float* presence_logits;       
    float* predicted_masks;       
};

/**
 * Thin TensorRT wrapper around the SAM3 mask decoder.
 * Owns output buffers. All input buffers are caller-owned.*/
class Sam3MaskDecoder : public Sam3ModelBase 
{
    public:
        explicit Sam3MaskDecoder(const std::string& plan_path);
        ~Sam3MaskDecoder() override;

        Sam3MaskDecoder(const Sam3MaskDecoder&) = delete;
        Sam3MaskDecoder& operator=(const Sam3MaskDecoder&) = delete;

        /**
         * Non-blocking forward pass of model.
         * @param image_features FPN features + positional encodings from Sam3ImageEncoder.
         * @param text_features Text embeddings from Sam3TextEncoder.
         * @param stream CUDA stream owned by Sam3Model.
         * @return Three device pointers to decoder outputs.
         * Caller syncs the stream before reading. */
        Sam3DecoderOutput decode(const Sam3ImageFeatures& image_features,
                                 const Sam3TextFeatures& text_features,
                                 const bool* d_attention_mask,
                                 const float* d_attention_mask_f,
                                 cudaStream_t stream);

    private:
        void discoverAndAllocate();

        float* d_predicted_logits_ = nullptr;
        float* d_predicted_boxes_ = nullptr;
        float* d_predicted_boxes_xyxy_ = nullptr;
        float* d_presence_logits_ = nullptr;
        float* d_predicted_masks_ = nullptr;
 
        Sam3DecoderOutput output_{};
};
