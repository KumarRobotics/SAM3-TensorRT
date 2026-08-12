#include "sam3tensorrt/Sam3MaskDecoder.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

Sam3MaskDecoder::Sam3MaskDecoder(const std::string& plan_path) : Sam3ModelBase(plan_path)
{
    discoverAndAllocate();
}

Sam3MaskDecoder::~Sam3MaskDecoder() 
{
    cudaFree(d_predicted_logits_);
    cudaFree(d_presence_logits_);
    cudaFree(d_predicted_masks_);
}

void Sam3MaskDecoder::discoverAndAllocate() 
{
    const int n = engine_->getNbIOTensors();

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            continue;  // all inputs are caller-owned, addresses set per-call in decode()

        const std::string sname(name);

        nvinfer1::Dims dims = engine_->getTensorShape(name);
        size_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d)
            count *= static_cast<size_t>(dims.d[d]);

        float** slot = nullptr;
        if (sname == "predicted_logits") {
            slot = &d_predicted_logits_;
        } else if (sname == "presence_logits") {
            slot = &d_presence_logits_;
        }else if (sname == "predicted_masks") {
            slot = &d_predicted_masks_;
        } else {
            throw std::runtime_error("Sam3MaskDecoder: unexpected output tensor: " + sname);
        }

        if (cudaMalloc(slot, count * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("Sam3MaskDecoder: cudaMalloc failed for " + sname);
        }

        context_->setTensorAddress(name, *slot);

        std::cout << "[Sam3MaskDecoder] output: " << name << " [" << count << " floats]\n";
    }

    output_.predicted_logits = d_predicted_logits_;
    output_.presence_logits  = d_presence_logits_;
    output_.predicted_masks  = d_predicted_masks_;
}

Sam3DecoderOutput Sam3MaskDecoder::decode(const Sam3ImageFeatures& image_features,
                                          const Sam3TextFeatures& text_features,
                                          const int32_t* d_attention_mask,
                                          const int32_t* d_original_sizes,
                                          cudaStream_t stream)
{
    for (int i = 0; i < 4; ++i) {
        const std::string fpn_name = "fpn_" + std::to_string(i);
        const std::string pos_name = "fpn_pos_" + std::to_string(i);
        context_->setTensorAddress(fpn_name.c_str(), image_features.fpn[i]);
        context_->setTensorAddress(pos_name.c_str(), image_features.fpn_pos[i]);
    }

    context_->setTensorAddress("text_embeds", text_features.text_embeddings);
    context_->setTensorAddress("attention_mask", const_cast<int32_t*>(d_attention_mask));
    context_->setTensorAddress("original_sizes", const_cast<int32_t*>(d_original_sizes));

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Sam3MaskDecoder: enqueueV3 failed");
    }

    return output_;
}
