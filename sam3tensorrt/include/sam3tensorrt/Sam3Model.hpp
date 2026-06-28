#pragma once

#include "sam3tensorrt/Sam3ImageEncoder.hpp"
#include "sam3tensorrt/Sam3TextEncoder.hpp"
#include "sam3tensorrt/Sam3MaskDecoder.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <string>

class Sam3Model
{
    public:
        Sam3Model(const std::string& models_path);
        ~Sam3Model();

        Sam3Model(const Sam3Model&) = delete;
        Sam3Model& operator=(const Sam3Model&) = delete;

        /**
         * Full forward pass: image + text --> masks.
         *
         * Image and text encoders run in parallel on separate streams.
         * Both are synced before the mask decoder is launched.
         * Blocks until all inference is complete.
         *
         * @param d_image           Preprocessed image [1, 3, 644, 644] NCHW float32.
         * @param d_input_ids       Token ids [1, 32] int32.
         * @param d_attention_mask  Attention mask [1, 32] int32. Shared with mask decoder.
         * @param d_original_sizes  Original image HW before resizing [1, 2] int32.
         * @return                  Device pointers to decoder outputs, owned by Sam3MaskDecoder.
         *                          Valid until the next forward() call or destruction. */
        Sam3DecoderOutput forward(const float* d_image, const int32_t* d_input_ids, const int32_t* d_attention_mask, const int32_t* d_original_sizes);

    private:
        Sam3ImageEncoder image_encoder_;
        Sam3TextEncoder  text_encoder_;
        Sam3MaskDecoder  mask_decoder_;

        cudaStream_t img_stream_ = nullptr;
        cudaStream_t txt_stream_ = nullptr;
        cudaStream_t decoder_stream_ = nullptr;
};
