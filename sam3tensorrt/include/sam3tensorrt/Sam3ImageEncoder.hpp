#pragma once

#include "sam3tensorrt/Sam3ModelBase.hpp"

#include <cuda_runtime.h>

struct Sam3ImageFeatures 
{
    float* fpn[4];      // fpn_0 .. fpn_3
    float* fpn_pos[4];  // fpn_pos_0 .. fpn_pos_3
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
         * @param stream   CUDA stream owned by Sam3Model.
         * @return         Struct of 8 device pointers. Caller syncs the stream before reading.*/
        Sam3ImageFeatures encode(const float* d_input, cudaStream_t stream);

    private:
        void discoverAndAllocate();

        float* d_fpn_[4]     = {};
        float* d_fpn_pos_[4] = {};

        Sam3ImageFeatures features_{};
};
