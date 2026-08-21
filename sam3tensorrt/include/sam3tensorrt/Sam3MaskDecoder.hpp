#pragma once

#include "sam3tensorrt/Sam3ModelBase.hpp"
#include "sam3tensorrt/Sam3ImageEncoder.hpp"
#include "sam3tensorrt/Sam3TextEncoder.hpp"

#include <string>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * Device pointers to mask decoder outputs.
 * Owned by Sam3MaskDecoder, valid until the next decode() call or destruction.
 * Caller must sync the stream before reading.*/
template <typename T>
struct Sam3DecoderOutput 
{
    T* predicted_logits; 
    T* predicted_boxes;       
    T* predicted_boxes_xyxy;  
    T* presence_logits;       
    T* predicted_masks;       
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
         * @param d_attention_mask txt_masks padding mask -- [1, 32] bool. 1 = pad, 0 = real token.
         * @param d_attention_mask_f Same as d_attention_mask, as float16.
         * @param stream CUDA stream owned by Sam3Model.
         * @return Three device pointers to decoder outputs.
         * Caller syncs the stream before reading. */
        Sam3DecoderOutput<float> decode(const Sam3ImageFeatures& image_features,
                                  const Sam3TextFeatures& text_features,
                                  const bool* d_attention_mask,
                                  const __half* d_attention_mask_f,
                                  cudaStream_t stream);

        Sam3DecoderOutput<float> toFloat(cudaStream_t stream);

    private:
        void discoverAndAllocate();

        __half* d_predicted_logits_ = nullptr;
        __half* d_predicted_boxes_ = nullptr;
        __half* d_predicted_boxes_xyxy_ = nullptr;
        __half* d_presence_logits_ = nullptr;
        __half* d_predicted_masks_ = nullptr;

        size_t predicted_logits_count_ = 0;
        size_t predicted_boxes_count_ = 0;
        size_t predicted_boxes_xyxy_count_ = 0;
        size_t presence_logits_count_ = 0;
        size_t predicted_masks_count_ = 0;
 
        float* d_predicted_logits_f_ = nullptr;
        float* d_predicted_boxes_f_ = nullptr;
        float* d_predicted_boxes_xyxy_f_ = nullptr;
        float* d_presence_logits_f_ = nullptr;
        float* d_predicted_masks_f_ = nullptr;

        Sam3DecoderOutput<__half> output_ {};
        Sam3DecoderOutput<float> output_f_ {};
};
