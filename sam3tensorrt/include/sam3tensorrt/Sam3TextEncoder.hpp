#pragma once

#include "sam3tensorrt/Sam3ModelBase.hpp"

#include <cstdint>
#include <cuda_runtime.h>

struct Sam3TextFeatures {
    float* text_embeddings;  // [1, 32, 256] float32
};

/**
 * Thin TensorRT wrapper around the SAM3 text encoder.
 * Owns the output buffer. Input buffers are caller-owned. Stream is caller-owned
 */
class Sam3TextEncoder : public Sam3ModelBase 
{
    public:
        explicit Sam3TextEncoder(const std::string& plan_path);
        ~Sam3TextEncoder() override;

        Sam3TextEncoder(const Sam3TextEncoder&)            = delete;
        Sam3TextEncoder& operator=(const Sam3TextEncoder&) = delete;

        /**
         * Non-blocking forward pass.
         * @param d_input_ids       input_ids on device -- [1, 32] int32.
         *                          Owned and written by Sam3Preprocessor.
         * @param d_attention_mask  attention_mask on device -- [1, 32] int32.
         *                          Owned and written by Sam3Preprocessor.
         * @param stream            CUDA stream owned by Sam3Model.
         * @return                  Device pointer to text_embeddings [1, 32, 256].
         *                          Caller syncs the stream before reading. */
        Sam3TextFeatures encode(const int32_t* d_input_ids,
                                const int32_t* d_attention_mask,
                                cudaStream_t stream);

    private:
        void discoverAndAllocate();

        float* d_text_embeddings_ = nullptr;

        Sam3TextFeatures features_{};
};
