cmake_minimum_required(VERSION 3.10)
project(SAM2-TRT)

# Set C++ standard
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-write-strings -Wall")
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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

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
