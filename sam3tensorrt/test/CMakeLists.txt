find_package(GTest REQUIRED)

add_executable(test_sam3model test_sam3model.cpp)
target_compile_definitions(test_sam3model PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_sam3model PRIVATE
    sam3tensorrt
    GTest::GTest
    GTest::Main
    CUDA::cudart)

add_executable(test_upsample test_upsample.cpp)
target_link_libraries(test_upsample PRIVATE
    sam3tensorrt_cuda
    GTest::GTest
    GTest::Main
    CUDA::cudart)

add_executable(test_timing test_timing.cpp)
target_compile_definitions(test_timing PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_timing PRIVATE
    sam3tensorrt
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(test_sam3model)
gtest_discover_tests(test_upsample)
gtest_discover_tests(test_timing)
