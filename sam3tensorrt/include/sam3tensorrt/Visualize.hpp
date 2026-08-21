#pragma once
 
#include "sam3tensorrt/Sam3Result.hpp"
 
#include <opencv2/opencv.hpp>

cv::Mat visualize(const cv::Mat& image, const Sam3Result& result, double mask_alpha = 0.45);
