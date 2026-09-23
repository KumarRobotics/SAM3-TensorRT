find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/include
)
find_library(TensorRT_nvinfer_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib
)
find_library(TensorRT_nvinfer_plugin_LIBRARY nvinfer_plugin
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvinfer_plugin_LIBRARY
)

if(TensorRT_FOUND)
    foreach(lib nvinfer nvinfer_plugin)
        if(NOT TARGET TensorRT::${lib})
            add_library(TensorRT::${lib} UNKNOWN IMPORTED)
            set_target_properties(TensorRT::${lib} PROPERTIES
                IMPORTED_LOCATION "${TensorRT_${lib}_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            )
        endif()
    endforeach()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvinfer_plugin_LIBRARY)
