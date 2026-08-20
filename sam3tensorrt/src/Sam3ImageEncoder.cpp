#include "sam3tensorrt/Sam3ImageEncoder.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

Sam3ImageEncoder::Sam3ImageEncoder(const std::string& plan_path) : Sam3ModelBase(plan_path)
{
    discoverAndAllocate();
}

Sam3ImageEncoder::~Sam3ImageEncoder() 
{
    cudaFree(d_input_native_);
    for (int i = 0; i < 3; ++i) {
        cudaFree(d_fpn_native_[i]);
        cudaFree(d_fpn_pos_native_[i]);
        cudaFree(d_fpn_[i]);
        cudaFree(d_fpn_pos_[i]);
    }
}

void Sam3ImageEncoder::discoverAndAllocate() 
{
    const int n = engine_->getNbIOTensors();

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_dtype_ = engine_->getTensorDataType(name);
            input_count_ = tensorCount(engine_->getTensorShape(name));

            if (cudaMalloc(&d_input_native_, input_count_ * dtypeSize(input_dtype_)) != cudaSuccess) {
                throw std::runtime_error("Sam3ImageEncoder: cudaMalloc failed for pixel_values");
            }
            continue;
        }

        const std::string sname(name);
        const std::string prefix = "output";
        if (sname.rfind(prefix, 0) != 0) {
            throw std::runtime_error("Sam3ImageEncoder: unexpected output tensor: " + sname);
        }
 
        const int output_num = std::stoi(sname.substr(prefix.size()));
 
        // output0-2 are FPN features, output3-5 are the corresponding position encodings
        const bool is_pos = output_num >= 3;
        const int idx = is_pos ? output_num - 3 : output_num;

        if (idx < 0 || idx > 2) {
            throw std::runtime_error("Sam3ImageEncoder: unexpected output tensor: " + sname);
        }

        nvinfer1::DataType dt = engine_->getTensorDataType(name);
        size_t count = tensorCount(engine_->getTensorShape(name));

        void** native_slot = is_pos ? &d_fpn_pos_native_[idx] : &d_fpn_native_[idx];
        __half** float_slot = is_pos ? &d_fpn_pos_[idx] : &d_fpn_[idx];
        nvinfer1::DataType* dtype_slot = is_pos ? &fpn_pos_dtype_[idx] : &fpn_dtype_[idx];
        size_t* count_slot = is_pos ? &fpn_pos_count_[idx] : &fpn_count_[idx];

        *dtype_slot = dt;
        *count_slot = count;

        // Native buffer
        if (cudaMalloc(native_slot, count * dtypeSize(dt)) != cudaSuccess) {
            throw std::runtime_error("Sam3ImageEncoder: cudaMalloc failed for " + sname);
        }
        // Float buffer
        if (cudaMalloc(float_slot, count * sizeof(float)) != cudaSuccess) {
            throw std::runtime_error("Sam3ImageEncoder: cudaMalloc failed for " + sname + " (float)");
        }

        context_->setTensorAddress(name, *native_slot);

        std::cout << "[Sam3ImageEncoder] output: " << name
                  << " [" << count << " elements, native dtype size " << dtypeSize(dt) << "B]\n";
    }

    context_->setTensorAddress("pixel_values", d_input_native_);

    for (int i = 0; i < 3; ++i) {
        features_.fpn[i] = d_fpn_[i];
        features_.fpn_pos[i] = d_fpn_pos_[i];
    }
}

Sam3ImageFeatures Sam3ImageEncoder::encode(const float* d_input, cudaStream_t stream) 
{
    convertToNative(d_input, d_input_native_, input_count_, input_dtype_, stream);

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Sam3ImageEncoder: enqueueV3 failed");
    }

    return features_;
}
