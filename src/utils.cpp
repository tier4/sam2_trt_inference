#include "utils.hpp"
#include "omp.h"


// 读取并转换坐标
std::vector<cv::Rect> read_and_transform_coordinates(const std::string &file_path)
{
    std::vector<cv::Rect> box_coords; // 存储所有的矩形
    std::ifstream file(file_path);    // 打开文件

    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file: " << file_path << std::endl;
        return box_coords;
    }

    std::string line;
    while (std::getline(file, line))
    { // 逐行读取文件
        std::istringstream iss(line);
        std::string part;
        std::vector<std::string> parts;

        // 按空格分割行内容
        while (iss >> part)
        {
            parts.push_back(part);
        }

        // 检查行内容是否符合格式要求
        if (parts.size() != 6)
        {
            continue; // 跳过不符合格式的行
        }

        // 提取坐标
        int x_min = std::stoi(parts[2]);
        int y_min = std::stoi(parts[3]);
        int x_max = std::stoi(parts[4]);
        int y_max = std::stoi(parts[5]);

        // 计算宽度和高度
        int width = x_max - x_min;
        int height = y_max - y_min;

        // 添加到结果列表
        box_coords.emplace_back(cv::Rect(x_min, y_min, width, height));
    }

    file.close();
    return box_coords;
}

// 生成随机颜色
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

// 绘制单个遮罩并添加边界
void draw_mask(cv::Mat &image, const cv::Mat &mask, const cv::Scalar &color, float alpha, bool draw_border)
{
    image.setTo(color, mask); // 黑色背景

    // 绘制轮廓
    if (draw_border)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::drawContours(image, contours, -1, color, 2); // -1 表示所有轮廓
    }
}

// 绘制多个遮罩
cv::Mat draw_masks(const cv::Mat &image, const std::vector<cv::Mat> &masks, float alpha, bool draw_border)
{
    cv::Mat mask_image = image.clone(); // 复制图像

    // 生成随机颜色
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
    cv::addWeighted(image, 1 - alpha, mask_image, alpha, 0, blended_image); // 透明度混合
    return blended_image;
}

//for debugging
void saveHighDimensionalArrayToCSV(const char* filename, const float* arr, const int* shape, int dims) {
    int total_size = 1;
    for (int i = 0; i < dims; ++i) {
        total_size *= shape[i];
    }

    std::ofstream file(filename);
    for (int idx = 0; idx < total_size; ++idx) {
        file << arr[idx];
        if ((idx + 1) % shape[dims - 1] == 0) { // 每一行结束
            file << "\n";
        } else {
            file << ",";
        }
    }
    file.close();
}

// 函数：保存 cv::Mat 到 CSV 文件
void saveMatToCSV(const cv::Mat &matrix, const std::string &filename) {
    // 打开输出文件
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // 遍历矩阵的每一行和每一列
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            // 输出矩阵的值到文件中
            file << matrix.at<float>(i, j); // 假设矩阵类型为 CV_32F
            if (j < matrix.cols - 1) {
                file << ","; // 用逗号分隔列
            }
        }
        file << "\n"; // 换行符分隔行
    }

    // 关闭文件
    file.close();
    std::cout << "Matrix saved to " << filename << std::endl;
}

// 保存 cv::Mat 到二进制文件
void saveBlobToBinary(const cv::Mat &blob, const std::string &filename) {
    // 打开二进制文件
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