find_package(GTest REQUIRED)

add_executable(test_sam3model test_sam3model.cpp)
target_compile_definitions(test_sam3model PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_sam3model PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    CUDA::cudart)

add_executable(test_upsample test_upsample.cpp)
target_link_libraries(test_upsample PRIVATE
    sam3trt_cuda
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    CUDA::cudart)

add_executable(test_timing test_timing.cpp)
target_compile_definitions(test_timing PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_timing PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    CUDA::cudart)

add_executable(test_image_encoder test_image_encoder.cpp)
target_compile_definitions(test_image_encoder PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_image_encoder PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    CUDA::cudart
)

add_executable(test_text_encoder test_text_encoder.cpp)
target_compile_definitions(test_text_encoder PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_text_encoder PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    CUDA::cudart
)

add_executable(test_mask_decoder test_mask_decoder.cpp)
target_compile_definitions(test_mask_decoder PRIVATE
    TEST_IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test.png"
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_mask_decoder PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    CUDA::cudart
)

add_executable(test_tokenizer test_tokenizer.cpp)
target_compile_definitions(test_tokenizer PRIVATE
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_tokenizer PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
)

add_executable(test_processor test_processor.cpp)
target_compile_definitions(test_processor PRIVATE
    SAM3_CONFIG_DIR="${CMAKE_SOURCE_DIR}/config")
target_link_libraries(test_processor PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    CUDA::cudart
)

add_executable(test_cuda_kernels test_cuda_kernels.cpp)
target_link_libraries(test_cuda_kernels PRIVATE
    sam3trt_cuda
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    CUDA::cudart
)

add_executable(test_visualize test_visualize.cpp)
target_link_libraries(test_visualize PRIVATE
    sam3trt
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(test_sam3model)
gtest_discover_tests(test_upsample)
gtest_discover_tests(test_timing)
gtest_discover_tests(test_image_encoder)
gtest_discover_tests(test_text_encoder)
gtest_discover_tests(test_mask_decoder)
gtest_discover_tests(test_tokenizer)
gtest_discover_tests(test_processor)
gtest_discover_tests(test_cuda_kernels)
gtest_discover_tests(test_visualize)
