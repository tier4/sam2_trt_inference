#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::vector<cv::Rect> ReadAndTransformCoordinates(const std::string &file_path);
std::vector<cv::Scalar> GenerateRandomColors(int num_colors, int seed = 2);
void DrawMask(cv::Mat &image, const cv::Mat &mask, const cv::Scalar &color,
              float alpha = 0.5, bool draw_border = false);
cv::Mat DrawMasks(const cv::Mat &image, const std::vector<cv::Mat> &masks,
                  float alpha = 0.5, bool draw_border = false);
std::string ReplaceOtherString(const std::string &str,
                               const std::string &old_str,
                               const std::string &new_str);
// for debugging
void SaveHighDimensionalArrayToCSV(const char *filename, const float *arr,
                                   const int *shape, int dims);
void SaveMatToCSV(const cv::Mat &matrix, const std::string &filename);
void SaveBlobToBinary(const cv::Mat &blob, const std::string &filename);