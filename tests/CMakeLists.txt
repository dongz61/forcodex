add_executable(quant_mul_mat "quant_mul_mat.cpp")
target_link_libraries(quant_mul_mat PRIVATE powerserve ggml)

if (POWERSERVE_WITH_QNN)
    add_executable(qnn_test qnn_test.cpp)
    target_link_libraries(qnn_test PRIVATE powerserve CLI11::CLI11)
endif()

if (POWERSERVE_WITH_OPENCL)
    # 首先尝试使用CMake的find_package
    find_package(OpenCL REQUIRED)
    
    if (OpenCL_FOUND)
        message(STATUS "OpenCL found:")
        message(STATUS "  Include dir: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "  Libraries: ${OpenCL_LIBRARIES}")
        message(STATUS "  Version: ${OpenCL_VERSION_STRING}")
        
        # 添加OpenCL测试程序
        add_executable(opencl_test "opencl_test.cpp")
        add_executable(matmul_diff_test "matmul_diff_test.cpp")
        add_executable(layer_diff_test "layer_diff_test.cpp")
        add_executable(softmax_ext_test "softmax_ext_test.cpp")
        add_executable(logits_cmp "logits_cmp.cpp")
        add_executable(tokens_cmp "tokens_cmp.cpp")
        
        # 链接OpenCL
        target_link_libraries(opencl_test PRIVATE 
            powerserve 
            OpenCL::OpenCL
        )
        target_link_libraries(matmul_diff_test PRIVATE
            powerserve
            OpenCL::OpenCL
        )
        target_link_libraries(layer_diff_test PRIVATE
            powerserve
            OpenCL::OpenCL
        )
        target_link_libraries(softmax_ext_test PRIVATE
            powerserve
            OpenCL::OpenCL
        )
        
        # 添加编译选项
        if (NOT MSVC)
            target_compile_options(opencl_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
            target_compile_options(matmul_diff_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
            target_compile_options(layer_diff_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
            target_compile_options(softmax_ext_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
        endif()

        target_link_libraries(logits_cmp PRIVATE
            powerserve
            OpenCL::OpenCL
        )

        if (NOT MSVC)
            target_compile_options(logits_cmp PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
        endif()

        target_include_directories(logits_cmp PRIVATE
            ${CMAKE_SOURCE_DIR}
        )

        target_link_libraries(tokens_cmp PRIVATE
            powerserve
            OpenCL::OpenCL
        )

        if (NOT MSVC)
            target_compile_options(tokens_cmp PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
        endif()

        target_include_directories(tokens_cmp PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        target_include_directories(matmul_diff_test PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        target_include_directories(layer_diff_test PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        target_include_directories(softmax_ext_test PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        
        # 添加一个方便运行测试的自定义命令
        add_custom_target(run_opencl_test
            COMMAND ./opencl_test
            DEPENDS opencl_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running OpenCL tests..."
        )
        add_custom_target(run_matmul_diff_test
            COMMAND POWERSERVE_USE_OPENCL=1 ./matmul_diff_test
            DEPENDS matmul_diff_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running matmul diff tests..."
        )
        add_custom_target(run_layer_diff_test
            COMMAND POWERSERVE_USE_OPENCL=1 ./layer_diff_test
            DEPENDS layer_diff_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running layer diff tests..."
        )
        add_custom_target(run_softmax_ext_test
            COMMAND POWERSERVE_USE_OPENCL=1 ./softmax_ext_test
            DEPENDS softmax_ext_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running softmax ext tests..."
        )

        add_custom_target(run_logits_cmp
            COMMAND POWERSERVE_USE_OPENCL=1 ./logits_cmp
            DEPENDS logits_cmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        
        add_custom_target(run_tokens_cmp
            COMMAND POWERSERVE_USE_OPENCL=1 ./tokens_cmp
            DEPENDS tokens_cmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

    else()
        message(FATAL_ERROR "OpenCL not found. Install OpenCL development package.")
    endif()
endif()
