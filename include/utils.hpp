#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp> 
#include <map>
#include <random>
#include <fstream>


std::vector<cv::Rect> read_and_transform_coordinates(const std::string &file_path);
std::vector<cv::Scalar> generate_random_colors(int num_colors, int seed = 2);
void draw_mask(cv::Mat& image, const cv::Mat& mask, const cv::Scalar& color, float alpha = 0.5, bool draw_border = false);
cv::Mat draw_masks(const cv::Mat& image, const std::vector<cv::Mat>& masks, float alpha = 0.5, bool draw_border = false);
std::string replaceOtherString(const std::string &str, const std::string &old_str, const std::string &new_str);
// for debugging
void saveHighDimensionalArrayToCSV(const char* filename, const float* arr, const int* shape, int dims);
void saveMatToCSV(const cv::Mat &matrix, const std::string &filename);
void saveBlobToBinary(const cv::Mat &blob, const std::string &filename);