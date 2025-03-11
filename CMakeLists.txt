cmake_minimum_required(VERSION 3.10)
project(SAM2-ORT)

# 设置 C++ 标准
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -O0 -g -Wno-write-strings -Wall")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=\$ORIGIN")

# 包含头文件目录
include_directories(/usr/local/include/onnxruntime)
include_directories(/usr/local/cuda/include include)

# 链接路径
link_directories(/usr/local/cuda/lib64)

# 依赖库
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib/onnxruntime)
find_package(OpenMP REQUIRED)

# 定义源文件
set(SOURCES
    src/main.cpp
    src/sam2_image_inference.cpp
    src/sam2_encoder.cpp
    src/sam2_decoder.cpp
    src/utils.cpp
)

# 定义可执行文件
add_executable(MyONNXGPUApp ${SOURCES})

# 链接库文件
target_link_libraries(MyONNXGPUApp PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBS} CUDA::cudart OpenMP::OpenMP_CXX)
target_include_directories(MyONNXGPUApp PRIVATE include)
