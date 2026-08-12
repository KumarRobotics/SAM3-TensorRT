#include "sam3tensorrt/Sam3TextEncoder.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

Sam3TextEncoder::Sam3TextEncoder(const std::string& plan_path) : Sam3ModelBase(plan_path)
{
    discoverAndAllocate();
}

Sam3TextEncoder::~Sam3TextEncoder() 
{
    cudaFree(d_text_embeddings_);
}

void Sam3TextEncoder::discoverAndAllocate()
{
    const int n = engine_->getNbIOTensors();

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            continue;
        }
        // Single output: text_embeddings [1, 32, 256]
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        size_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            count *= static_cast<size_t>(dims.d[d]);
        }

        if (cudaMalloc(&d_text_embeddings_, count * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("Sam3TextEncoder: cudaMalloc failed for text_embeddings");
        }

        context_->setTensorAddress(name, d_text_embeddings_);

        std::cout << "[Sam3TextEncoder] output: " << name << " [" << count << " floats]\n";
    }

    features_.text_embeddings = d_text_embeddings_;
}

Sam3TextFeatures Sam3TextEncoder::encode(const int32_t* d_input_ids, const int32_t* d_attention_mask, cudaStream_t stream)
{
    context_->setTensorAddress("input_ids", const_cast<int32_t*>(d_input_ids));
    context_->setTensorAddress("attention_mask", const_cast<int32_t*>(d_attention_mask));

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Sam3TextEncoder: enqueueV3 failed");
    }

    return features_;
}
