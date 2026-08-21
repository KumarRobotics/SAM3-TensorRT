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
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

include_directories(
    include
    ${TRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

add_library(sam3tensorrt_cuda STATIC 
  src/normalize.cu 
  src/preprocess.cu
  src/upsample.cu
  src/typing.cu
)
target_include_directories(sam3tensorrt_cuda PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(sam3tensorrt_cuda PROPERTIES
  CUDA_ARCHITECTURES 
  ${CMAKE_CUDA_ARCHITECTURES}
)

add_library(${PROJECT_NAME} STATIC
    src/Sam3ModelBase.cpp
    src/Sam3ImageEncoder.cpp
    src/Sam3TextEncoder.cpp
    src/Sam3MaskDecoder.cpp
    src/Sam3Processor.cpp
    src/Sam3Model.cpp
    src/Tokenizer.cpp
    src/Visualize.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  sam3tensorrt_cuda
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  ${OpenCV_LIBS}
  ${JSONCPP_LIBRARIES}
  CUDA::cudart
)

add_executable(sam3 sam3.cpp)
target_compile_definitions(sam3 PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
)
target_link_libraries(sam3 PRIVATE
  sam3tensorrt
  ${OpenCV_LIBS}
)

set_target_properties(sam3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

enable_testing()
add_subdirectory(test)
