cmake_minimum_required(VERSION 3.24)
project(sam3tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures. 'native' auto-detects at build time. \
Override with a semicolon-separated list for GPU-less CI builds e.g. '86;89;90'.")

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/include
    REQUIRED
)

find_library(NVINFER_LIB nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${NVINFER_LIB}")

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
    include
)

add_library(sam3_cuda STATIC 
  src/normalize.cu
)
target_include_directories(sam3_cuda PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(sam3_cuda PROPERTIES
  CUDA_ARCHITECTURES 
  ${CMAKE_CUDA_ARCHITECTURES}
)

add_library(${PROJECT_NAME} STATIC
    src/Sam3ModelBase.cpp
    src/Sam3ImageEncoder.cpp
    src/Sam3TextEncoder.cpp
    src/Sam3MaskDecoder.cpp
    src/Sam3Model.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${TRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  sam3_cuda
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  ${OpenCV_LIBS}
  CUDA::cudart
)


