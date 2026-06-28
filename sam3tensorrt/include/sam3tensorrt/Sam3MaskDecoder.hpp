#pragma once

#include "sam3tensorrt/Sam3ModelBase.hpp"
#include "sam3tensorrt/Sam3ImageEncoder.hpp"
#include "sam3tensorrt/Sam3TextEncoder.hpp"

#include <string>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * Device pointers to mask decoder outputs.
 * Owned by Sam3MaskDecoder — valid until the next decode() call or destruction.
 * Caller must sync the stream before reading.*/
struct Sam3DecoderOutput {
    float* predicted_logits;
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
         * Non-blocking forward pass.
         * @param image_features    FPN features + positional encodings from Sam3ImageEncoder.
         *                          image_features.fpn[i] → fpn_i
         *                          image_features.pos[i] → fpn_pos_i  (name differs by design)
         * @param text_features     Text embeddings from Sam3TextEncoder.
         *                          text_features.text_embeddings → text_embeds
         * @param d_attention_mask  [1, 32] int32 — same buffer used by Sam3TextEncoder,
         *                          owned by Sam3Preprocessor.
         * @param d_original_sizes  [1, 2]  int32 — original image HW before resizing,
         *                          owned by Sam3Preprocessor.
         * @param stream            CUDA stream owned by Sam3Model.
         * @return                  Three device pointers to decoder outputs.
         *                          Caller syncs the stream before reading. */
        Sam3DecoderOutput decode(const Sam3ImageFeatures& image_features,
                                 const Sam3TextFeatures& text_features,
                                 const int32_t* d_attention_mask,
                                 const int32_t* d_original_sizes,
                                 cudaStream_t stream);

    private:
        void discoverAndAllocate();

        float* d_predicted_logits_ = nullptr;
        float* d_presence_logits_ = nullptr;
        float* d_predicted_masks_ = nullptr;

        Sam3DecoderOutput output_{};
};
