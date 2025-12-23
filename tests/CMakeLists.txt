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
        
        # 链接OpenCL
        target_link_libraries(opencl_test PRIVATE 
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
        endif()
        
        # 添加一个方便运行测试的自定义命令
        add_custom_target(run_opencl_test
            COMMAND ./opencl_test
            DEPENDS opencl_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running OpenCL tests..."
        )
        
    else()
        message(FATAL_ERROR "OpenCL not found. Install OpenCL development package.")
    endif()
endif()