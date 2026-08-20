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
    cudaFree(d_text_features_native_);
    cudaFree(d_text_embeddings_native_);
    cudaFree(d_text_features_);
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

        const std::string sname(name);

        void** native_slot = nullptr;
        __half** float16_slot = nullptr;
        nvinfer1::DataType* dtype_slot = nullptr;
        size_t* count_slot = nullptr;

        if (sname == "output0") {
            native_slot = &d_text_features_native_;
            float16_slot  = &d_text_features_;
            dtype_slot  = &text_features_dtype_;
            count_slot  = &text_features_count_;
        } else if (sname == "output2") {
            native_slot = &d_text_embeddings_native_;
            float16_slot  = &d_text_embeddings_;
            dtype_slot  = &text_embeddings_dtype_;
            count_slot  = &text_embeddings_count_;
        } else {
            throw std::runtime_error("Sam3TextEncoder: unexpected output tensor: " + sname);
        }

        nvinfer1::DataType dt = engine_->getTensorDataType(name);
        size_t count = tensorCount(engine_->getTensorShape(name));

        *dtype_slot = dt;
        *count_slot = count;

        // Native buffer: TensorRT writes here directly, sized by the engine's actual dtype.
        if (cudaMalloc(native_slot, count * dtypeSize(dt)) != cudaSuccess) {
            throw std::runtime_error("Sam3TextEncoder: cudaMalloc failed for " + sname);
        }
        // Float buffer: what encode() hands back to callers, converted from native after every run.
        if (cudaMalloc(float16_slot, count * sizeof(__half)) != cudaSuccess) {
            throw std::runtime_error("Sam3TextEncoder: cudaMalloc failed for " + sname + " (float)");
        }

        context_->setTensorAddress(name, *native_slot);

        std::cout << "[Sam3TextEncoder] output: " << name
                  << " [" << count << " elements, native dtype size " << dtypeSize(dt) << "B]\n";
    }

    features_.text_features = d_text_features_;
    features_.text_embeddings = d_text_embeddings_;
}

Sam3TextFeatures Sam3TextEncoder::encode(const int64_t* d_input_ids, const bool* d_attention_mask, cudaStream_t stream)
{
    context_->setTensorAddress("input_ids", const_cast<int64_t*>(d_input_ids));
    context_->setTensorAddress("output1", const_cast<bool*>(d_attention_mask));

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Sam3TextEncoder: enqueueV3 failed");
    }

    return features_;
}
