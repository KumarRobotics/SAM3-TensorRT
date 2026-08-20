#include "sam3tensorrt/cuda/typing.cuh"

#include <cuda_fp16.h>
#include <stdexcept>

namespace {

__global__ void floatToHalfKernel(const float* in, __half* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void halfToFloatKernel(const __half* in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

}

void floatToNative(const float* in, void* out, size_t n, nvinfer1::DataType dt, cudaStream_t s) 
{
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            cudaMemcpyAsync(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice, s);
            break;
        case nvinfer1::DataType::kHALF:
            floatToHalfKernel<<<blocks, threads, 0, s>>>(in, static_cast<__half*>(out), n);
            break;
        default:
            throw std::runtime_error("Sam3ModelBase: unsupported native dtype (float->native)");
    }
}

void nativeToFloat(const void* in, float* out, size_t n, nvinfer1::DataType dt, cudaStream_t s) 
{
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            cudaMemcpyAsync(out, in, n * sizeof(float), cudaMemcpyDeviceToDevice, s);
            break;
        case nvinfer1::DataType::kHALF:
            halfToFloatKernel<<<blocks, threads, 0, s>>>(static_cast<const __half*>(in), out, n);
            break;
        default:
            throw std::runtime_error("Sam3ModelBase: unsupported native dtype (native->float)");
    }
}
