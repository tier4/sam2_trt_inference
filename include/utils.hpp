/**
 * @file utils.hpp
 * @brief Utility functions for SAM2 image processing
 *
 * Copyright (c) 2024 TIERIV
 * Author: Hunter Cheng (haoxuan.cheng@tier4.jp)
 * Created: 2025.4
 */

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
std::vector<cv::Rect> read_and_transform_coordinates(const std::string& file_path);

// Generate random colors for visualization
std::vector<cv::Scalar> generate_random_colors(int num_colors, int seed = 2);

// Draw single mask on image with optional border
void draw_mask(cv::Mat& image,
               const cv::Mat& mask,
               const cv::Scalar& color,
               float alpha = 0.5,
               bool draw_border = false);

// Draw multiple masks on image with optional borders
cv::Mat draw_masks(const cv::Mat& image,
                   const std::vector<cv::Mat>& masks,
                   float alpha = 0.5,
                   bool draw_border = false);

// Replace substring in string
std::string replaceOtherString(const std::string& str,
                               const std::string& old_str,
                               const std::string& new_str);

// Debugging utilities
void saveHighDimensionalArrayToCSV(const char* filename,
                                   const float* arr,
                                   const int* shape,
                                   int dims);
void saveMatToCSV(const cv::Mat& matrix, const std::string& filename);
void saveBlobToBinary(const cv::Mat& blob, const std::string& filename);