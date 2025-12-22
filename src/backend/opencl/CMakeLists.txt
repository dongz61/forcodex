# src/backend/opencl/CMakeLists.txt
# 查找OpenCL库
find_package(OpenCL QUIET)

if (OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
    
    # 添加OpenCL源文件到主库
    target_sources(powerserve PRIVATE
        "opencl_backend.cpp"
        "opencl_context.cpp"
        "opencl_memory.cpp"
        "opencl_kernel_manager.cpp"
        "opencl_buffer.cpp"
        # 后续添加更多文件
    )
    
    # 添加包含目录
    target_include_directories(powerserve PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCL_INCLUDE_DIRS}
    )
    
    # 链接OpenCL库
    target_link_libraries(powerserve PRIVATE ${OpenCL_LIBRARIES})
    
    # 添加编译定义
    target_compile_definitions(powerserve PRIVATE
        GGML_USE_OPENCL
        POWERSERVE_WITH_OPENCL
    )
    
    # 嵌入式内核支持（可选）
    if (POWERSERVE_OPENCL_EMBED_KERNELS)
        message(STATUS "OpenCL kernels will be embedded into binary")
        
        # 添加编译定义
        target_compile_definitions(powerserve PRIVATE
            POWERSERVE_OPENCL_EMBED_KERNELS
        )
        
        # 查找Python
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        if (Python3_Interpreter_FOUND)
            message(STATUS "Python found: ${Python3_EXECUTABLE}")
            
            # 创建generated目录
            set(OPENCL_KERNELS_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
            file(MAKE_DIRECTORY ${OPENCL_KERNELS_GEN_DIR})
            
            # 扫描kernels目录
            file(GLOB OPENCL_KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cl")
            set(OPENCL_EMBEDDED_HEADERS "")
            
            foreach(cl_file ${OPENCL_KERNEL_FILES})
                # 获取文件名（不含扩展名）
                get_filename_component(KERNEL_NAME ${cl_file} NAME_WE)
                
                # 设置输入输出文件路径
                set(INPUT_FILE ${cl_file})
                set(OUTPUT_FILE "${OPENCL_KERNELS_GEN_DIR}/${KERNEL_NAME}.cl.h")
                
                # 为每个内核定义宏
                string(TOUPPER ${KERNEL_NAME} KERNEL_NAME_UPPER)
                target_compile_definitions(powerserve PRIVATE
                    OPENCL_${KERNEL_NAME_UPPER}_CL_AVAILABLE
                )
                
                # 添加自定义命令生成头文件
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
                
                # 收集生成的头文件
                list(APPEND OPENCL_EMBEDDED_HEADERS ${OUTPUT_FILE})
            endforeach()
            
            # 添加生成的目录到包含路径
            target_include_directories(powerserve PRIVATE ${OPENCL_KERNELS_GEN_DIR})
            
            # 添加依赖
            add_custom_target(opencl_kernels_generated DEPENDS ${OPENCL_EMBEDDED_HEADERS})
            add_dependencies(powerserve opencl_kernels_generated)
            
        else()
            message(WARNING "Python not found. Cannot embed OpenCL kernels.")
            set(POWERSERVE_OPENCL_EMBED_KERNELS OFF CACHE BOOL "Embed OpenCL kernels" FORCE)
        endif()
    else()
        message(STATUS "OpenCL kernels will be loaded from files at runtime")
    endif()
    
    if (POWERSERVE_OPENCL_DEBUG)
        target_compile_definitions(powerserve PRIVATE
            GGML_OPENCL_DEBUG
        )
    endif()
    
    if (POWERSERVE_OPENCL_PROFILING)
        target_compile_definitions(powerserve PRIVATE
            GGML_OPENCL_PROFILING
        )
    endif()
    
else()
    message(WARNING "OpenCL not found. OpenCL backend will be disabled.")
    # 可选：将POWERSERVE_WITH_OPENCL设置为OFF
    set(POWERSERVE_WITH_OPENCL OFF CACHE BOOL "Enable OpenCL backend" FORCE)
endif()