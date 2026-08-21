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
    cudaFree(d_text_features_);
    cudaFree(d_text_embeddings_);
}

void Sam3TextEncoder::discoverAndAllocate()
{
    const int n = engine_->getNbIOTensors();

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            // input_ids, and the tensor confusingly named "output1" in the
            // plan that actually carries attention_mask. Both are caller-owned
            // and already in native dtype (int64 / bool) via Sam3Processor,
            // bound directly in encode() -- nothing to allocate here.
            continue;
        }

        const std::string sname(name);

        __half** slot = nullptr;
        size_t* count_slot = nullptr;

        if (sname == "output0") {
            slot = &d_text_features_;
            count_slot = &text_features_count_;
        } else if (sname == "output2") {
            slot = &d_text_embeddings_;
            count_slot = &text_embeddings_count_;
        } else {
            throw std::runtime_error("Sam3TextEncoder: unexpected output tensor: " + sname);
        }

        nvinfer1::DataType dt = engine_->getTensorDataType(name);
        if (dt != nvinfer1::DataType::kHALF) {
            // This wrapper hands the decoder raw fp16 pointers with no
            // conversion step, so a non-fp16 binding here is a hard error
            // rather than something to silently coerce.
            throw std::runtime_error("Sam3TextEncoder: expected fp16 binding for " + sname);
        }

        size_t count = tensorCount(engine_->getTensorShape(name));
        *count_slot = count;

        if (cudaMalloc(slot, count * sizeof(__half)) != cudaSuccess) {
            throw std::runtime_error("Sam3TextEncoder: cudaMalloc failed for " + sname);
        }

        context_->setTensorAddress(name, *slot);

        std::cout << "[Sam3TextEncoder] output: " << name << " [" << count << " fp16 elements]\n";
    }

    features_.text_features = d_text_features_;
    features_.text_embeddings = d_text_embeddings_;
}

Sam3TextFeatures Sam3TextEncoder::encode(const int64_t* d_input_ids, const bool* d_attention_mask, cudaStream_t stream)
{
    // Caller already provides these in the engine's native dtype -- bind directly.
    context_->setTensorAddress("input_ids", const_cast<int64_t*>(d_input_ids));
    context_->setTensorAddress("output1", const_cast<bool*>(d_attention_mask));

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Sam3TextEncoder: enqueueV3 failed");
    }
    // enqueueV3 returning true only means the work was queued -- it does not
    // guarantee the kernels completed without error. That only surfaces via
    // the next sync's return code or cudaGetLastError(). Caller must check.

    // Outputs are left in native fp16 -- the decoder consumes them directly.
    return features_;
}
