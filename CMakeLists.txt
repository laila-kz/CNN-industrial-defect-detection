cmake_minimum_required(VERSION 3.18)
project(CNNIndustrialDefectsDetection 
    VERSION 1.0.0 
    DESCRIPTION "High-performance C++ & LibTorch Industrial Defect Detection System"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 1. DEPENDENCY DISCOVERY (OpenCV & LibTorch)
# ==============================================================================

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library found: ${OpenCV_INCLUDE_DIRS} (Version: ${OpenCV_VERSION})")

# Determine Torch_DIR dynamically if not explicitly specified via CLI (-DTorch_DIR=...)
if(NOT Torch_DIR AND NOT DEFINED ENV{Torch_DIR} AND NOT DEFINED ENV{TORCH_DIR})
    if(EXISTS "C:/libtorch/libtorch/share/cmake/Torch")
        set(Torch_DIR "C:/libtorch/libtorch/share/cmake/Torch" CACHE PATH "Path to LibTorch CMake package")
    elseif(EXISTS "$ENV{LOCALAPPDATA}/libtorch/share/cmake/Torch")
        set(Torch_DIR "$ENV{LOCALAPPDATA}/libtorch/share/cmake/Torch" CACHE PATH "Path to LibTorch CMake package")
    endif()
endif()

# Find LibTorch
find_package(Torch REQUIRED)

if(Torch_FOUND)
    message(STATUS "LibTorch found: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "LibTorch install prefix: ${TORCH_INSTALL_PREFIX}")
    if(Torch_CUDA_FOUND)
        message(STATUS "CUDA acceleration enabled for LibTorch (CUDA Version: ${TORCH_CUDA_VERSION})")
    else()
        message(STATUS "CUDA not detected in LibTorch bundle. Running in CPU mode.")
    endif()
else()
    message(FATAL_ERROR "LibTorch not found. Please set -DTorch_DIR=<path-to-libtorch>/share/cmake/Torch")
endif()

# ==============================================================================
# 2. SOURCE SPECIFICATIONS
# ==============================================================================

set(COMMON_SOURCES
    src/ImageLoader.cpp
    src/Preprocessor.cpp
    src/CNNModel.cpp
    src/DecisionEngine.cpp
    src/Evaluator.cpp
    src/Visualizer.cpp
    src/DataLoader.cpp
    src/LossFunctions.cpp
    src/ModelTrainer.cpp
)

# Main Inference Executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/ImageLoader.cpp
    src/Preprocessor.cpp
    src/DecisionEngine.cpp
    src/Evaluator.cpp
    src/CNNModel.cpp
    src/Visualizer.cpp
)

# Training Executable
add_executable(CNNIndustrialDefectsTraining
    src/train_main.cpp
    src/ModelTrainer.cpp
    src/DataLoader.cpp
    src/LossFunctions.cpp
    src/ImageLoader.cpp
    src/Preprocessor.cpp
    src/CNNModel.cpp
)

# ==============================================================================
# 3. INCLUDE DIRECTORIES & COMPILATION FLAGS
# ==============================================================================

foreach(TARGET_NAME ${PROJECT_NAME} CNNIndustrialDefectsTraining)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
    )

    target_compile_definitions(${TARGET_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        USE_TORCH
    )

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W3 /MP /bigobj)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pthread)
    endif()
endforeach()

# ==============================================================================
# 4. RUNTIME ASSET & SHARED LIBRARY MANAGEMENT
# ==============================================================================

# Ensure config files are copied to the build destination
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/config)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config/config.yaml
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/training_config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config/training_config.yaml
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    COPYONLY
)

# On Windows, deploy required dynamic link libraries (DLLs) adjacent to binaries
if(WIN32)
    set(TORCH_DLL_DIR "")
    if(DEFINED TORCH_INSTALL_PREFIX AND EXISTS "${TORCH_INSTALL_PREFIX}/lib")
        set(TORCH_DLL_DIR "${TORCH_INSTALL_PREFIX}/lib")
    elseif(EXISTS "C:/libtorch/libtorch/lib")
        set(TORCH_DLL_DIR "C:/libtorch/libtorch/lib")
    endif()

    if(TORCH_DLL_DIR)
        file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")
        foreach(TARGET_NAME ${PROJECT_NAME} CNNIndustrialDefectsTraining)
            foreach(dll ${TORCH_DLLS})
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll}"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                    COMMENT "Deploying ${dll} to target directory"
                )
            endforeach()
        endforeach()
    endif()

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release recommended for LibTorch)" FORCE)
    endif()
endif()