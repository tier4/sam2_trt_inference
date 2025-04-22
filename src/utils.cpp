/**
 * @file utils.cpp
 * @brief Utility functions for SAM2 image processing
 * 
 * Copyright (c) 2024 TIERIV
 * Author: Hunter Cheng (haoxuan.cheng@tier4.jp)
 * Created: 2025.4
 */

#include "utils.hpp"
#include "omp.h"


// Read and transform coordinates from file
std::vector<cv::Rect> read_and_transform_coordinates(const std::string &file_path)
{
    std::vector<cv::Rect> box_coords; // Store all rectangles
    std::ifstream file(file_path);    // Open file

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file: " << file_path << std::endl;
        return box_coords;
    }

    std::string line;
    while (std::getline(file, line))
    { // Read file line by line
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
            continue; // Skip lines that don't match format
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
std::vector<cv::Scalar> generate_random_colors(int num_colors, int seed)
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
void draw_mask(cv::Mat &image, const cv::Mat &mask, const cv::Scalar &color, float alpha, bool draw_border)
{
    image.setTo(color, mask); // Black background

    // Draw contour
    if (draw_border)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::drawContours(image, contours, -1, color, 2); // -1 means all contours
    }
}

// Draw multiple masks
cv::Mat draw_masks(const cv::Mat &image, const std::vector<cv::Mat> &masks, float alpha, bool draw_border)
{
    cv::Mat mask_image = image.clone(); // Copy image

    // Generate random colors
    auto colors = generate_random_colors(200, 42);

    #pragma omp parallel for
    for (size_t i = 0; i < masks.size(); i++)
    {
        if (masks[i].empty())
            continue;

        cv::Scalar color = colors[i % colors.size()];
        draw_mask(mask_image, masks[i], color, alpha, draw_border);
    }
    cv::Mat blended_image;
    cv::addWeighted(image, 1 - alpha, mask_image, alpha, 0, blended_image); // Alpha blending
    return blended_image;
}

std::string replaceOtherString(const std::string &str, const std::string &old_str, const std::string &new_str) {
    std::string result = str;
    size_t start_pos = result.find(old_str);
    if(start_pos == std::string::npos) {
        return result; // Return original string if old_str not found
    }
    return result.replace(start_pos, old_str.length(), new_str);
}

// For debugging
void saveHighDimensionalArrayToCSV(const char* filename, const float* arr, const int* shape, int dims) {
    int total_size = 1;
    for (int i = 0; i < dims; ++i) {
        total_size *= shape[i];
    }

    std::ofstream file(filename);
    for (int idx = 0; idx < total_size; ++idx) {
        file << arr[idx];
        if ((idx + 1) % shape[dims - 1] == 0) { // End of each row
            file << "\n";
        } else {
            file << ",";
        }
    }
    file.close();
}

// Save cv::Mat to CSV file
void saveMatToCSV(const cv::Mat &matrix, const std::string &filename) {
    // Open output file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Iterate through each row and column of the matrix
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            // Output matrix value to file
            file << matrix.at<float>(i, j); // Assuming matrix type is CV_32F
            if (j < matrix.cols - 1) {
                file << ","; // Separate columns with comma
            }
        }
        file << "\n"; // Separate rows with newline
    }

    // Close file
    file.close();
    std::cout << "Matrix saved to " << filename << std::endl;
}

// Save cv::Mat to binary file
void saveBlobToBinary(const cv::Mat &blob, const std::string &filename) {
    // Open binary file
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        return;
    }

    if (blob.isContinuous()) {
        file.write(reinterpret_cast<const char *>(blob.data), blob.total() * blob.elemSize());
    } else {
        std::cerr << "Matrix is not continuous!" << std::endl;
    }
    file.close();
    std::cout << "Blob saved to " << filename << std::endl;
}