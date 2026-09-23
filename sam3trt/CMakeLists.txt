cmake_minimum_required(VERSION 3.24)
project(sam3trt VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures. 'native' auto-detects at build time. \
Override with a semicolon-separated list for GPU-less CI builds e.g. '86;89;90'.")

option(SAM3TRT_BUILD_TESTS "Build the gtest unit tests" ON)
option(SAM3TRT_BUILD_ROS "Build the ROS 2 node when ROS 2 packages are available" ON)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)

set(SAM3TRT_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_DATADIR}/sam3trt/config")
set(SAM3TRT_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/sam3trt")

add_library(sam3trt_cuda STATIC
  src/normalize.cu
  src/preprocess.cu
  src/upsample.cu
  src/typing.cu
)
target_include_directories(sam3trt_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(sam3trt_cuda PUBLIC
  TensorRT::nvinfer
  CUDA::cudart
)
set_target_properties(sam3trt_cuda PROPERTIES
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  POSITION_INDEPENDENT_CODE ON
)

add_library(sam3trt STATIC
    src/Sam3ModelBase.cpp
    src/Sam3ImageEncoder.cpp
    src/Sam3TextEncoder.cpp
    src/Sam3MaskDecoder.cpp
    src/Sam3Processor.cpp
    src/Sam3Model.cpp
    src/Tokenizer.cpp
    src/Visualize.cpp
)
target_include_directories(sam3trt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(sam3trt PUBLIC
  sam3trt_cuda
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin
  ${OpenCV_LIBS}
  PkgConfig::JSONCPP
  CUDA::cudart
)
set_target_properties(sam3trt PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(sam3 sam3.cpp)
target_compile_definitions(sam3 PRIVATE
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config"
    SAM3_INSTALL_CONFIG_DIR="${SAM3TRT_INSTALL_CONFIG_DIR}"
)
target_link_libraries(sam3 PRIVATE
  sam3trt
  ${OpenCV_LIBS}
)
set_target_properties(sam3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

install(TARGETS sam3trt sam3trt_cuda
    EXPORT sam3trtTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(TARGETS sam3 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/sam3trt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY config/ DESTINATION ${SAM3TRT_INSTALL_CONFIG_DIR})

install(EXPORT sam3trtTargets
    NAMESPACE sam3trt::
    DESTINATION ${SAM3TRT_INSTALL_CMAKE_DIR}
)
configure_package_config_file(
    cmake/sam3trtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sam3trtConfig.cmake
    INSTALL_DESTINATION ${SAM3TRT_INSTALL_CMAKE_DIR}
    PATH_VARS SAM3TRT_INSTALL_CONFIG_DIR
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sam3trtConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sam3trtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sam3trtConfigVersion.cmake
    cmake/FindTensorRT.cmake
    DESTINATION ${SAM3TRT_INSTALL_CMAKE_DIR}
)

if(SAM3TRT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(SAM3TRT_BUILD_ROS)
    foreach(pkg ament_cmake ament_index_cpp rclcpp rclcpp_components sensor_msgs std_msgs)
        find_package(${pkg} QUIET)
        if(NOT ${pkg}_FOUND)
            message(STATUS "${pkg} not found, skipping ROS node")
            set(SAM3TRT_BUILD_ROS OFF)
            break()
        endif()
    endforeach()
endif()

if(SAM3TRT_BUILD_ROS)
    add_library(sam3_component SHARED ros/Sam3Node.cpp)
    target_link_libraries(sam3_component PUBLIC
        sam3trt
        ament_index_cpp::ament_index_cpp
        rclcpp::rclcpp
        rclcpp_components::component
        ${sensor_msgs_TARGETS}
        ${std_msgs_TARGETS}
    )
    rclcpp_components_register_nodes(sam3_component "Sam3Node")

    add_executable(sam3_node ros/node.cpp)
    target_link_libraries(sam3_node PRIVATE sam3_component)

    install(TARGETS sam3_component LIBRARY DESTINATION lib)
    install(TARGETS sam3_node RUNTIME DESTINATION lib/${PROJECT_NAME})
    install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

    ament_package()
endif()
