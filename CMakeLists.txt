cmake_minimum_required(VERSION 3.18)  # Increased for LibTorch compatibility
project(CNNIndustrialDefectsDetection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# 1. FIND PACKAGES
# ============================================

# Find OpenCV
find_package(OpenCV REQUIRED)

# Tell CMake where to find LibTorch
set(Torch_DIR "C:/libtorch/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Optional: Check if CUDA is available
if(Torch_FOUND)
    message(STATUS "LibTorch found: ${TORCH_INCLUDE_DIRS}")
    if(Torch_CUDA_FOUND)
        message(STATUS "CUDA available: ${TORCH_CUDA_VERSION}")
    else()
        message(STATUS "CUDA not available, using CPU")
    endif()
endif()

# ============================================
# 2. ADD EXECUTABLES
# ============================================

# Main inference executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/ImageLoader.cpp
    src/Preprocessor.cpp
    src/DecisionEngine.cpp
    src/Evaluator.cpp
    src/CNNModel.cpp
    src/Visualizer.cpp
)

# Training executable
add_executable(CNNIndustrialDefectsTraining
    src/train_main.cpp
    src/ModelTrainer.cpp
    src/DataLoader.cpp
    src/LossFunctions.cpp
    src/ImageLoader.cpp
    src/Preprocessor.cpp
    src/CNNModel.cpp
)

# ============================================
# 3. INCLUDE DIRECTORIES
# ============================================

# For inference executable
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# For training executable
target_include_directories(CNNIndustrialDefectsTraining PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# ============================================
# 4. LINK LIBRARIES
# ============================================

# For inference executable
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# For training executable
target_link_libraries(CNNIndustrialDefectsTraining PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# ============================================
# 5. WINDOWS-SPECIFIC SETTINGS
# ============================================

if(WIN32)
    # Copy LibTorch DLLs to output directory
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/libtorch/libtorch/lib/torch.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/libtorch/libtorch/lib/c10.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/libtorch/libtorch/lib/torch_cpu.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
    
    # Copy all other DLLs (simpler approach)
    file(GLOB TORCH_DLLS "C:/libtorch/libtorch/lib/*.dll")
    foreach(dll ${TORCH_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach()
    
    # Set release mode by default (LibTorch is release build)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# ============================================
# 6. COMPILE DEFINITIONS
# ============================================

# Add necessary definitions for LibTorch (both executables)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
)

target_compile_definitions(CNNIndustrialDefectsTraining PRIVATE
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
)

# ============================================
# 7. ADD CONFIG FILE TO OUTPUT
# ============================================

# Copy config.yaml to output directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    COPYONLY
)