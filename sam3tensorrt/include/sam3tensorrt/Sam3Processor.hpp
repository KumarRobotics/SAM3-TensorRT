/*!
* @author Jason Hughes
* @date June 2026 
*
* @about object to wrap cuda and non-cuda pre and post processing 
* operations
*/

#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "sam3tensorrt/cuda/preprocess.cuh"

namespace Sam3TensorRT 
{

struct Sam3Parameters
{
    int   image_size;        // Target square image size (default 1008)
    float image_mean[3];     // Per-channel mean  {R, G, B}
    float image_std[3];      // Per-channel std   {R, G, B}
    int   text_padding;      // Max token length (default 77)
    float score_threshold;   // Minimum final score to keep a mask
    float mask_threshold;    // Sigmoid threshold for binary mask

    Sam3Parameters() = default;
    Sam3Parameters(const std::string& yaml_path);
};

class Sam3Processor
{
    Sam3Processor() = default;
    Sam3Processor(const std::string& yaml_path,
                  const std::string& merges_path,
                  const std::string& vocab_path);
};

} // namespace Sam3TensorRT
