#pragma once

#include "sam3tensorrt/Tokenizer.hpp"
#include "sam3tensorrt/cuda/typing.cuh"
#include "sam3tensorrt/cuda/normalize.cuh"

#include <cuda_fp6.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <cstring>

struct Sam3ImageInput 
{
    float* d_image; // [1, 3, 644, 644] NCHW float32
    int32_t* d_original_sizes;  // [1, 2] int32  {height, width} before resize
};

struct Sam3TextInput
{
    int64_t* d_input_ids; // [1, 32] int32
    bool* d_attention_mask;  // [1, 32] padding mask: 1 = pad, 0 = real token
    __half* d_attention_mask_f; // [1, 32] same, as float16
};

struct Sam3Input
{
    Sam3ImageInput image;
    std::vector<Sam3TextInput> text;
};

class Sam3Processor 
{
    public:
        struct Params 
        {
            int image_size = 644;
            int text_padding = 32;
            float image_mean[3] = {0.485f, 0.456f, 0.406f};  // RGB order
            float image_std[3] = {0.229f, 0.224f, 0.225f};  // RGB order
        };

        Sam3Processor(const std::string& merges_path, const std::string& vocab_path, const Params& params);
        Sam3Processor(const std::string& merges_path, const std::string& vocab_path);
        ~Sam3Processor();

        Sam3Processor(const Sam3Processor&) = delete;
        Sam3Processor& operator=(const Sam3Processor&) = delete;

        /**
         * Resize to [644,644] on CPU, upload to GPU, normalize in CUDA.
         * Captures original image dimensions for the mask decoder.
         *
         * @param image BGR uint8 cv::Mat, any resolution.
         * @return Device pointers valid until the next preprocessImage() call*/
        Sam3ImageInput preprocessImage(const cv::Mat& image);

        /**
         * Tokenize, pad to 32, upload to GPU.
         *
         * @param text Raw input string.
         * @return Device pointers valid until the next preprocessText() call.*/
        Sam3TextInput preprocessText(const std::string& text);

        Sam3Input preprocess(const cv::Mat& image, const std::vector<std::string>& texts);

    private:
        void allocateBuffers();

        void uploadImage(const cv::Mat& resized);
        void uploadOriginalSizes(int h, int w);
        void uploadText(const std::vector<int64_t>& ids, const std::vector<uint8_t>& mask, const std::vector<__half>& mask_f);

        Params params_;
        CLIPTokenizer tokenizer_;
        cudaStream_t stream_ = nullptr;

        uint8_t* h_bgr_ = nullptr; // pinned host staging [644,644,3] uint8
        uint8_t* d_bgr_ = nullptr; // device source [644,644,3] uint8
        float* d_image_ = nullptr; // device output [1,3,644,644] float32

        int32_t* h_original_sizes_ = nullptr; // pinned host [2] int32
        int32_t* d_original_sizes_ = nullptr; // device [1, 2] int32

        int64_t* h_input_ids_ = nullptr;  // pinned host 32 int32
        uint8_t* h_attention_mask_ = nullptr;  // pinned host 32 int32
        __half* h_attention_mask_f_ = nullptr;
        
        int64_t* d_input_ids_ = nullptr;  // device [1, 32] int32
        bool* d_attention_mask_ = nullptr;  // device [1, 32] int32
        __half* d_attention_mask_f_ = nullptr;
};
