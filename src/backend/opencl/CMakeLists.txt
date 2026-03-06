# src/backend/opencl/CMakeLists.txt

# 如果未启用 OpenCL，则直接跳过该后端
if (NOT POWERSERVE_WITH_OPENCL)
    message(STATUS "POWERSERVE_WITH_OPENCL is OFF. OpenCL backend will be disabled.")
    return()
endif()

message(STATUS "Configuring OpenCL backend sources")

# 1) 注册 OpenCL 后端源码
target_sources(powerserve PRIVATE
    "opencl_backend.cpp"
    "opencl_kernels_ops.cpp"
    "opencl_kv_cache_ops.cpp"
    "opencl_context.cpp"
    "opencl_memory.cpp"
    "opencl_kernel_manager.cpp"
    "opencl_buffer.cpp"
    "opencl_tensor_ops.cpp"
)

# 2) 添加头文件目录和编译宏
target_include_directories(powerserve PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(powerserve PRIVATE
    GGML_USE_OPENCL
    POWERSERVE_WITH_OPENCL
)

# 3) 可选：将 OpenCL kernel 嵌入二进制
if (POWERSERVE_OPENCL_EMBED_KERNELS)
    message(STATUS "OpenCL kernels will be embedded into binary")

    target_compile_definitions(powerserve PRIVATE
        POWERSERVE_OPENCL_EMBED_KERNELS
    )

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if (Python3_Interpreter_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE}")

        set(OPENCL_KERNELS_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
        file(MAKE_DIRECTORY ${OPENCL_KERNELS_GEN_DIR})

        file(GLOB OPENCL_KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cl")
        set(OPENCL_EMBEDDED_HEADERS "")

        foreach(cl_file ${OPENCL_KERNEL_FILES})
            get_filename_component(KERNEL_NAME ${cl_file} NAME_WE)

            set(INPUT_FILE ${cl_file})
            set(OUTPUT_FILE "${OPENCL_KERNELS_GEN_DIR}/${KERNEL_NAME}.cl.h")

            string(TOUPPER ${KERNEL_NAME} KERNEL_NAME_UPPER)
            target_compile_definitions(powerserve PRIVATE
                OPENCL_${KERNEL_NAME_UPPER}_CL_AVAILABLE
            )

            add_custom_command(
                OUTPUT ${OUTPUT_FILE}
                COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_SOURCE_DIR}/tools/embed_opencl_kernel.py
                        ${INPUT_FILE}
                        ${OUTPUT_FILE}
                DEPENDS ${INPUT_FILE}
                        ${CMAKE_SOURCE_DIR}/tools/embed_opencl_kernel.py
                COMMENT "Generating embedded OpenCL kernel: ${KERNEL_NAME}"
            )

            list(APPEND OPENCL_EMBEDDED_HEADERS ${OUTPUT_FILE})
        endforeach()

        target_include_directories(powerserve PRIVATE ${OPENCL_KERNELS_GEN_DIR})

        add_custom_target(opencl_kernels_generated DEPENDS ${OPENCL_EMBEDDED_HEADERS})
        add_dependencies(powerserve opencl_kernels_generated)
    else()
        message(WARNING "Python not found. Cannot embed OpenCL kernels.")
        set(POWERSERVE_OPENCL_EMBED_KERNELS OFF CACHE BOOL "Embed OpenCL kernels" FORCE)
    endif()
else()
    message(STATUS "OpenCL kernels will be loaded from files at runtime")
endif()

# 4) 链接 OpenCL 运行库
if (ANDROID)
    if (DEFINED POWERSERVE_ANDROID_OPENCL_LIB AND EXISTS "${POWERSERVE_ANDROID_OPENCL_LIB}")
        message(STATUS "Using Android OpenCL library: ${POWERSERVE_ANDROID_OPENCL_LIB}")
        target_link_libraries(powerserve PRIVATE "${POWERSERVE_ANDROID_OPENCL_LIB}")
    else()
        message(WARNING
            "Android build: OpenCL library not provided. "
            "Set -DPOWERSERVE_ANDROID_OPENCL_LIB=/abs/path/to/libOpenCL.so "
            "(arm64) to link successfully."
        )
    endif()
else()
    # 非 Android 平台通过 find_package(OpenCL) 查找系统 OpenCL
    find_package(OpenCL QUIET)
    if (OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
        target_include_directories(powerserve PRIVATE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(powerserve PRIVATE ${OpenCL_LIBRARIES})
    else()
        message(WARNING "OpenCL not found. You enabled POWERSERVE_WITH_OPENCL but OpenCL cannot be linked on this platform.")
    endif()
endif()
