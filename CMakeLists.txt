cmake_minimum_required(VERSION 3.10)
project(SAM2-TRT)

# Let users specify -DCMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

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

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include header directories
include_directories(/usr/local/cuda/include include)

# Link directories
link_directories(/usr/local/cuda/lib64)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TensorRT REQUIRED)
find_package(nlohmann_json REQUIRED)

# Define source files
set(SOURCES
  src/main.cpp
  src/sam2_image_inference.cpp
  src/sam2_encoder.cpp
  src/sam2_decoder.cpp
  src/sam2_ctypes.cpp
  src/utils.cpp
  src/colormap.cpp
  src/tensorrt_common/simple_profiler.cpp
  src/tensorrt_common/tensorrt_common.cpp
)

# Create shared library for Python bindings
add_library(trt_sam2_infer SHARED ${SOURCES})
target_link_libraries(trt_sam2_infer PRIVATE 
  ${OpenCV_LIBS}
  CUDA::cudart
  OpenMP::OpenMP_CXX
  nvinfer
  nvinfer_plugin
  nvparsers
  nvonnxparser
  cuda
  cudart
  boost_system
  boost_filesystem
  dl
  "stdc++fs"
  "stdc++"
  nlohmann_json::nlohmann_json
)
target_include_directories(trt_sam2_infer PRIVATE include)

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
  boost_system
  boost_filesystem
  dl
  "stdc++fs"
  "stdc++"
)
target_include_directories(trtsam2 PRIVATE include)

# Install targets
install(TARGETS trt_sam2_infer
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS trtsam2
  RUNTIME DESTINATION bin
)

# Install Python package
install(DIRECTORY python/
  DESTINATION lib/python3.8/site-packages/trt_sam2
  FILES_MATCHING PATTERN "*.py"
)
