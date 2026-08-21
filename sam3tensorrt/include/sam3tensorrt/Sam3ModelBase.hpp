#pragma once
 
#include <memory>
#include <string>
 
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "sam3tensorrt/cuda/typing.cuh"
/**
 * Shared base for all SAM3 TensorRT sub-models.
 * Owns the TRT runtime/engine/context and handles plan loading.
 * Subclasses implement discoverAndAllocate() and their forward pass.
 */
class Sam3ModelBase 
{
    protected:
        explicit Sam3ModelBase(const std::string& plan_path);
        virtual ~Sam3ModelBase() = default;
     
        Sam3ModelBase(const Sam3ModelBase&) = delete;
        Sam3ModelBase& operator=(const Sam3ModelBase&) = delete;
    
        static size_t dtypeSize(nvinfer1::DataType dt);
        static size_t tensorCount(const nvinfer1::Dims& dims);
        void convertToNative(const float* in, void* out, size_t n, nvinfer1::DataType dt, cudaStream_t s);
        void convertFromNative(const void* in, float* out, size_t n, nvinfer1::DataType dt, cudaStream_t s);
        
        struct Logger : public nvinfer1::ILogger 
        {
            void log(Severity severity, const char* msg) noexcept override;
        };
     
        Logger logger_;
     
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
     
    private:
        void loadPlan(const std::string& path);
};
