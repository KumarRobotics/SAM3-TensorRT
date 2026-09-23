#include "sam3trt/Visualize.hpp"
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
 
namespace {
 
// Deterministic per-class color palette (BGR)
cv::Scalar color_for_class_id(int class_id) {
    static const std::vector<cv::Scalar> palette = {
        {56, 56, 255},   {255, 122, 56},  {56, 219, 255},
        {107, 255, 56},  {255, 56, 220},  {56, 255, 149},
        {255, 191, 56},  {143, 56, 255},  {255, 56, 87},
        {56, 149, 255}
    };
    return palette[class_id % static_cast<int>(palette.size())];
}
 
void draw_mask(cv::Mat& output, const cv::Mat& mask, const cv::Scalar& color, double mask_alpha) {
    if (mask.empty()) {
        return;
    }
 
    cv::Mat mask_resized;
    if (mask.size() != output.size()) {
        cv::resize(mask, mask_resized, output.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        mask_resized = mask;
    }
 
    cv::Mat mask_8u;
    if (mask_resized.type() != CV_8UC1) {
        mask_resized.convertTo(mask_8u, CV_8UC1);
    } else {
        mask_8u = mask_resized;
    }
 
    cv::Mat color_mask(output.size(), output.type(), color);
    cv::Mat blended;
    cv::addWeighted(output, 1.0 - mask_alpha, color_mask, mask_alpha, 0.0, blended);
    blended.copyTo(output, mask_8u);
 
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(output, contours, -1, color, 1, cv::LINE_AA);
}
 
void draw_bbox_and_label(cv::Mat& output, const Detection& det, const cv::Scalar& color) {
    if (det.bbox.size() != 4) {
        return;
    }
 
    const int x = det.bbox[0];
    const int y = det.bbox[1];
    const int w = det.bbox[2];
    const int h = det.bbox[3];
 
    cv::Rect box(x, y, w, h);
    box &= cv::Rect(0, 0, output.cols, output.rows); // clip to image bounds
 
    cv::rectangle(output, box, color, 2, cv::LINE_AA);
 
    char conf_buf[16];
    std::snprintf(conf_buf, sizeof(conf_buf), "%.2f", det.confidence);
    const std::string label = det.class_label + " " + conf_buf;
 
    int baseline = 0;
    const double font_scale = 0.5;
    const int thickness = 1;
    const cv::Size text_size = cv::getTextSize(
        label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
 
    const int label_y = std::max(box.y, text_size.height + 4);
    const cv::Point text_origin(box.x, label_y - baseline);
 
    cv::rectangle(
        output,
        cv::Point(box.x, label_y - text_size.height - baseline - 4),
        cv::Point(box.x + text_size.width + 4, label_y),
        color, cv::FILLED);
 
    cv::putText(
        output, label, cv::Point(box.x + 2, label_y - baseline - 2),
        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
        thickness, cv::LINE_AA);
}
 
}  // namespace
 
cv::Mat visualize(const cv::Mat& image, const Sam3Result& result, double mask_alpha) {
    if (image.empty()) {
        throw std::invalid_argument("draw_sam3_result: input image is empty");
    }
 
    cv::Mat output = image.clone();
    if (output.channels() == 1) {
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    }
 
    for (const Detection& det : result.detections) {
        const cv::Scalar color = color_for_class_id(det.class_id);
        draw_mask(output, det.mask, color, mask_alpha); 
        draw_bbox_and_label(output, det, color);
    }
 
    return output;
}
