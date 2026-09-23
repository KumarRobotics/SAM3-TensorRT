#pragma once

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

void floatToNative(const float* in, void* out, size_t n, nvinfer1::DataType dt, cudaStream_t s);

void nativeToFloat(const void* in, float* out, size_t n, nvinfer1::DataType dt, cudaStream_t s);
