add_executable(kv_cache_persist_test "kv_cache_persist_test.cpp")
target_link_libraries(kv_cache_persist_test PRIVATE powerserve)

if (NOT MSVC)
    target_compile_options(kv_cache_persist_test PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
    )
endif()

target_include_directories(kv_cache_persist_test PRIVATE
    ${CMAKE_SOURCE_DIR}
)

add_custom_target(run_kv_cache_persist_test
    COMMAND ./kv_cache_persist_test
    DEPENDS kv_cache_persist_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running KV cache persist test..."
)

add_executable(qwen2_segmented_cmp "qwen2_segmented_cmp.cpp")
target_link_libraries(qwen2_segmented_cmp PRIVATE powerserve)

if (NOT MSVC)
    target_compile_options(qwen2_segmented_cmp PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
    )
endif()

target_include_directories(qwen2_segmented_cmp PRIVATE
    ${CMAKE_SOURCE_DIR}
)

add_custom_target(run_qwen2_segmented_cmp
    COMMAND ./qwen2_segmented_cmp
    DEPENDS qwen2_segmented_cmp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running qwen2 segmented compare..."
)

if (POWERSERVE_WITH_QNN)
    add_executable(qnn_test qnn_test.cpp)
    target_link_libraries(qnn_test PRIVATE powerserve CLI11::CLI11)
endif()

if (POWERSERVE_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    if (OpenCL_FOUND)
        add_executable(layer_diff_test "layer_diff_test.cpp")
        add_executable(logits_cmp "logits_cmp.cpp")

        target_link_libraries(layer_diff_test PRIVATE
            powerserve
            OpenCL::OpenCL
        )

        target_link_libraries(logits_cmp PRIVATE
            powerserve
            OpenCL::OpenCL
        )

        if (NOT MSVC)
            target_compile_options(layer_diff_test PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
            target_compile_options(logits_cmp PRIVATE
                -Wall
                -Wextra
                -Wno-unused-parameter
            )
        endif()

        target_include_directories(layer_diff_test PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
        target_include_directories(logits_cmp PRIVATE
            ${CMAKE_SOURCE_DIR}
        )

        add_custom_target(run_layer_diff_test
            COMMAND POWERSERVE_USE_OPENCL=1 ./layer_diff_test
            DEPENDS layer_diff_test
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running layer diff tests..."
        )

        add_custom_target(run_logits_cmp
            COMMAND POWERSERVE_USE_OPENCL=1 ./logits_cmp
            DEPENDS logits_cmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running logits compare tests..."
        )
    endif()
endif()

