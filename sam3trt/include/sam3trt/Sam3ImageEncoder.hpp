#pragma once

#include "sam3trt/Sam3ModelBase.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct Sam3ImageFeatures 
{
    __half* fpn[3];      // fpn_0 .. fpn_2
    __half* fpn_pos[3];  // fpn_pos_0 .. fpn_pos_2
};

class Sam3ImageEncoder : public Sam3ModelBase 
{
    public:
        explicit Sam3ImageEncoder(const std::string& plan_path);
        ~Sam3ImageEncoder() override;

        Sam3ImageEncoder(const Sam3ImageEncoder&)            = delete;
        Sam3ImageEncoder& operator=(const Sam3ImageEncoder&) = delete;

        /**
         * Non-blocking forward pass.
         * @param d_input  Preprocessed image on device — [1, 3, H, W] NCHW Owned and written by Sam3Processor.
         * @param stream  CUDA stream owned by Sam3Model.
         * @return Struct of 8 device pointers. Caller syncs the stream before reading.*/
        Sam3ImageFeatures encode(const float* d_input, cudaStream_t stream);

    private:
        void discoverAndAllocate();

        void* d_input_native_ = nullptr;

        nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;

        size_t input_count_ = 0;
        size_t fpn_count_[3] = {};
        size_t fpn_pos_count_[3] = {};

        __half* d_fpn_[3] = {};
        __half* d_fpn_pos_[3] = {};

        Sam3ImageFeatures features_{};
};
