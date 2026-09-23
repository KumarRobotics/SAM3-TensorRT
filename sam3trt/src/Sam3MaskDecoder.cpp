#include "sam3trt/Sam3MaskDecoder.hpp"

#include <iostream>
#include <istream>
#include <stdexcept>
#include <string>

Sam3MaskDecoder::Sam3MaskDecoder(const std::string& plan_path) : Sam3ModelBase(plan_path)
{
    discoverAndAllocate();
}

Sam3MaskDecoder::~Sam3MaskDecoder() 
{
    cudaFree(d_predicted_logits_);
    cudaFree(d_predicted_boxes_);
    cudaFree(d_predicted_boxes_xyxy_);
    cudaFree(d_presence_logits_);
    cudaFree(d_predicted_masks_);

    cudaFree(d_predicted_logits_f_);
    cudaFree(d_predicted_boxes_f_);
    cudaFree(d_predicted_boxes_xyxy_f_);
    cudaFree(d_presence_logits_f_);
    cudaFree(d_predicted_masks_f_);
}

void Sam3MaskDecoder::discoverAndAllocate() 
{
    const int n = engine_->getNbIOTensors();
 
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
 
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            continue; 
        }
 
        const std::string sname(name);
 
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        size_t count = tensorCount(dims);
 
        __half** slot = nullptr;
        float** float_slot = nullptr;
        size_t* count_slot = nullptr;
 
        if (sname == "output0") {
            slot = &d_predicted_logits_;
            float_slot = &d_predicted_logits_f_;
            count_slot = &predicted_logits_count_;
        } else if (sname == "output1") {
            slot = &d_predicted_boxes_;
            float_slot = &d_predicted_boxes_f_;
            count_slot = &predicted_boxes_count_;
        } else if (sname == "output2") {
            slot = &d_predicted_boxes_xyxy_;
            float_slot = &d_predicted_boxes_xyxy_f_;
            count_slot = &predicted_boxes_xyxy_count_;
        } else if (sname == "output3") {
            slot = &d_presence_logits_;
            float_slot = &d_presence_logits_f_;
            count_slot = &presence_logits_count_;
        } else if (sname == "output4") {
            slot = &d_predicted_masks_;
            float_slot = &d_predicted_masks_f_;
            count_slot = &predicted_masks_count_;
        } else {
            throw std::runtime_error("Sam3MaskDecoder: unexpected output tensor: " + sname);
        }
 
        *count_slot = count;
 
        if (cudaMalloc(slot, count * sizeof(__half)) != cudaSuccess) {
            throw std::runtime_error("Sam3MaskDecoder: cudaMalloc failed for " + sname);
        }
        if (cudaMalloc(float_slot, count * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("Sam3MaskDecoder: cudaMalloc failed for " + sname + " (float)");
        }
 
        context_->setTensorAddress(name, *slot);
 
        std::cout << "[Sam3MaskDecoder] output: " << name << " [" << count << " fp16 elements]\n";
    }
 
    output_.predicted_logits = d_predicted_logits_;
    output_.predicted_boxes = d_predicted_boxes_;
    output_.predicted_boxes_xyxy = d_predicted_boxes_xyxy_;
    output_.presence_logits = d_presence_logits_;
    output_.predicted_masks = d_predicted_masks_;
 
    output_f_.predicted_logits = d_predicted_logits_f_;
    output_f_.predicted_boxes = d_predicted_boxes_f_;
    output_f_.predicted_boxes_xyxy = d_predicted_boxes_xyxy_f_;
    output_f_.presence_logits = d_presence_logits_f_;
    output_f_.predicted_masks = d_predicted_masks_f_;
}

Sam3DecoderOutput<float> Sam3MaskDecoder::decode(const Sam3ImageFeatures& image_features,
                                          const Sam3TextFeatures& text_features,
                                          const bool* d_attention_mask,
                                          const __half* d_txt_masks_f,
                                          cudaStream_t stream)
{
    for (int i = 0; i < 3; ++i) {
        const std::string fpn_name = "fpn" + std::to_string(i);
        const std::string pos_name = "pos" + std::to_string(i);
        context_->setTensorAddress(fpn_name.c_str(), image_features.fpn[i]);
        context_->setTensorAddress(pos_name.c_str(), image_features.fpn_pos[i]);
    }


    context_->setTensorAddress("txt_feats", text_features.text_features);

    context_->setTensorAddress("txt_masks", const_cast<bool*>(d_attention_mask));
    context_->setTensorAddress("txt_masks_f", const_cast<__half*>(d_txt_masks_f));

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Sam3MaskDecoder: enqueueV3 failed");
    }

    return toFloat(stream);
}

Sam3DecoderOutput<float> Sam3MaskDecoder::toFloat(cudaStream_t stream)
{
    convertFromNative(d_predicted_logits_, d_predicted_logits_f_, predicted_logits_count_, nvinfer1::DataType::kHALF, stream);
    convertFromNative(d_predicted_boxes_, d_predicted_boxes_f_, predicted_boxes_count_, nvinfer1::DataType::kHALF, stream);
    convertFromNative(d_predicted_boxes_xyxy_, d_predicted_boxes_xyxy_f_, predicted_boxes_xyxy_count_, nvinfer1::DataType::kHALF, stream);
    convertFromNative(d_presence_logits_, d_presence_logits_f_, presence_logits_count_, nvinfer1::DataType::kHALF, stream);
    convertFromNative(d_predicted_masks_, d_predicted_masks_f_, predicted_masks_count_, nvinfer1::DataType::kHALF, stream);
 
    return output_f_;
}
