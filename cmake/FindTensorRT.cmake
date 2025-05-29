# Copyright 2025 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FindTensorRT.cmake
#
# Finds the TensorRT library
#
# This will define the following variables:
#
#   TensorRT_FOUND        - True if the system has TensorRT
#   TensorRT_INCLUDE_DIRS - TensorRT include directory
#   TensorRT_LIBRARIES    - TensorRT libraries
#   TensorRT_VERSION      - TensorRT version

include(FindPackageHandleStandardArgs)

# Find TensorRT include directory
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
    /usr/local/TensorRT/include
  PATH_SUFFIXES
    tensorrt
)

# Find TensorRT libraries
find_library(TensorRT_LIBRARY
  NAMES nvinfer
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/TensorRT/lib
)

find_library(TensorRT_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/TensorRT/lib
)

find_library(TensorRT_PARSER_LIBRARY
  NAMES nvparsers
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/TensorRT/lib
)

find_library(TensorRT_ONNX_PARSER_LIBRARY
  NAMES nvonnxparser
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/TensorRT/lib
)

# Set TensorRT_FOUND
set(TensorRT_LIBRARIES
  ${TensorRT_LIBRARY}
  ${TensorRT_PLUGIN_LIBRARY}
  ${TensorRT_PARSER_LIBRARY}
  ${TensorRT_ONNX_PARSER_LIBRARY}
)

set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

# Try to get version
if(TensorRT_INCLUDE_DIR)
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_VERSION_MAJOR
    REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_VERSION_MINOR
    REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_VERSION_PATCH
    REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1"
    TensorRT_VERSION_MAJOR "${TensorRT_VERSION_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1"
    TensorRT_VERSION_MINOR "${TensorRT_VERSION_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1"
    TensorRT_VERSION_PATCH "${TensorRT_VERSION_PATCH}")

  set(TensorRT_VERSION
    "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION
)

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_LIBRARY
  TensorRT_PLUGIN_LIBRARY
  TensorRT_PARSER_LIBRARY
  TensorRT_ONNX_PARSER_LIBRARY
) 