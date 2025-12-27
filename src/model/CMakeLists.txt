add_subdirectory(llama)
add_subdirectory(internvl)
add_subdirectory(qwen2)
add_subdirectory(module)
add_subdirectory(qwen3)

target_sources(powerserve PRIVATE
    "model_loader.cpp"
)
