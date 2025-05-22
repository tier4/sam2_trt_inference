// Copyright 2025 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Read and transform coordinates from file
std::vector<cv::Rect> ReadAndTransformCoordinates(const std::string& file_path);

// Generate random colors for visualization
std::vector<cv::Scalar> GenerateRandomColors(int num_colors, int seed = 2);

// Draw single mask on image with optional border
void DrawMask(cv::Mat& image,
               const cv::Mat& mask,
               const cv::Scalar& color,
               float alpha = 0.5,
               bool draw_border = false);

// Draw multiple masks on image with optional borders
cv::Mat DrawMasks(const cv::Mat& image,
                   const std::vector<cv::Mat>& masks,
                   float alpha = 0.5,
                   bool draw_border = false);

// Replace substring in string
std::string ReplaceFileExtension(const std::string& str,
                               const std::string& old_str,
                               const std::string& new_str);

// Debugging utilities
void saveHighDimensionalArrayToCSV(const char* filename,
                                   const float* arr,
                                   const int* shape,
                                   int dims);
void saveMatToCSV(const cv::Mat& matrix, const std::string& filename);
void saveBlobToBinary(const cv::Mat& blob, const std::string& filename);