cmake_minimum_required(VERSION 3.10)
project(SAM2-TRT)

# Set build type option
option(BUILD_TYPE "Build type (Debug/Release)" Release)
set(CMAKE_BUILD_TYPE ${BUILD_TYPE})

# Set C++ standard
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Set compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -Wall -Wextra -Wno-write-strings")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
    message(STATUS "Building in Debug mode")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-write-strings -Wall")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    message(STATUS "Building in Release mode")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=\$ORIGIN")

# Include header directories
include_directories(/usr/local/cuda/include include)

# Link directories
link_directories(/usr/local/cuda/lib64)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# Define source files
set(SOURCES
    src/main.cpp
    src/sam2_image_inference.cpp
    src/sam2_encoder.cpp
    src/sam2_decoder.cpp
    src/utils.cpp
    src/tensorrt_common/simple_profiler.cpp
    src/tensorrt_common/tensorrt_common.cpp
)

# Define executable
add_executable(trtsam2 ${SOURCES})

# Link libraries
target_link_libraries(trtsam2 PRIVATE 
    ${OpenCV_LIBS}
    CUDA::cudart
    OpenMP::OpenMP_CXX
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
    cuda
    cudart
    gflags
    boost_system
    boost_filesystem
    dl
    "stdc++fs"
    "stdc++"
)
target_include_directories(trtsam2 PRIVATE include)
