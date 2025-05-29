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



#include "utils.hpp"

#include "omp.h"

// Read and transform coordinates from file
std::vector<cv::Rect> ReadAndTransformCoordinates(const std::string& file_path)
{
    std::vector<cv::Rect> box_coords;  // Store all rectangles
    std::ifstream file(file_path);     // Open file

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file: " << file_path << std::endl;
        return box_coords;
    }

    std::string line;
    while (std::getline(file, line))
    {  // Read file line by line
        std::istringstream iss(line);
        std::string part;
        std::vector<std::string> parts;

        // Split line content by spaces
        while (iss >> part)
        {
            parts.push_back(part);
        }

        // Check if line content matches required format
        if (parts.size() != 6)
        {
            continue;  // Skip lines that don't match format
        }

        // Extract coordinates
        int x_min = std::stoi(parts[2]);
        int y_min = std::stoi(parts[3]);
        int x_max = std::stoi(parts[4]);
        int y_max = std::stoi(parts[5]);

        // Calculate width and height
        int width = x_max - x_min;
        int height = y_max - y_min;

        // Add to result list
        box_coords.emplace_back(cv::Rect(x_min, y_min, width, height));
    }

    file.close();
    return box_coords;
}

// Generate random colors
std::vector<cv::Scalar> GenerateRandomColors(int num_colors, int seed)
{
    std::vector<cv::Scalar> colors;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> dist(0, 255);

    for (int i = 0; i < num_colors; ++i)
    {
        colors.emplace_back(cv::Scalar(dist(rng), dist(rng), dist(rng)));
    }

    return colors;
}

// Draw single mask and add border
void DrawMask(cv::Mat& image,
               const cv::Mat& mask,
               const cv::Scalar& color,
               float alpha,
               bool draw_border)
{
    image.setTo(color, mask);  // Black background

    // Draw contour
    if (draw_border)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::drawContours(image, contours, -1, color, 2);  // -1 means all contours
    }
}

// Draw multiple masks
cv::Mat DrawMasks(const cv::Mat& image,
                   const std::vector<cv::Mat>& masks,
                   float alpha,
                   bool draw_border)
{
    cv::Mat mask_image = image.clone();  // Copy image

    // Generate random colors
    auto colors = GenerateRandomColors(200, 42);

#pragma omp parallel for
    for (size_t i = 0; i < masks.size(); i++)
    {
        if (masks[i].empty())
            continue;

        cv::Scalar color = colors[i % colors.size()];
        DrawMask(mask_image, masks[i], color, alpha, draw_border);
    }
    cv::Mat blended_image;
    cv::addWeighted(image, 1 - alpha, mask_image, alpha, 0, blended_image);  // Alpha blending
    return blended_image;
}

std::string ReplaceFileExtension(const std::string& str,
                               const std::string& old_str,
                               const std::string& new_str)
{
    std::string result = str;
    size_t start_pos = result.find(old_str);
    if (start_pos == std::string::npos)
    {
        return result;  // Return original string if old_str not found
    }
    return result.replace(start_pos, old_str.length(), new_str);
}

std::vector<std::vector<cv::Point>> get_polygons( const cv::Mat &mask)
{  
  std::vector<std::vector<cv::Point>> contours;  
  //cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  contours.reserve(1000);
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  return contours;
}