#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class SAM2ImageEncoder
{
public:
    // 构造函数
    SAM2ImageEncoder(const std::string &path, const std::string &model_precision);

    // 对图像进行编码
    void EncodeImage(const std::vector<cv::Mat> &images);

    // 编码后的高分辨率特征
    const Ort::Float16_t *feats_0_data;
    const Ort::Float16_t *feats_1_data;
    const Ort::Float16_t *embed_data;

    // 输入和输出的名称
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // 输入的宽高
    int input_height_;
    int input_width_;

    // 模型的精度（fp16 或 fp32）
    int model_precision_;

    // 批处理大小
    int batch_size_;

private:
    // ONNX 运行时环境
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "SAM2ImageEncoder"};

    // ONNX 运行时会话
    Ort::Session session_{nullptr};

    // 内存分配器
    Ort::AllocatorWithDefaultOptions allocator;

    // 从 ONNX 模型获取输入的详细信息
    void GetInputDetails();

    // 从 ONNX 模型获取输出的详细信息
    void GetOutputDetails();

    // 准备输入张量
    cv::Mat PrepareInput(const std::vector<cv::Mat> &images);

    // 执行推理
    std::vector<Ort::Value> Infer(const cv::Mat &input_tensor);

    // 处理推理输出
    void ProcessOutput(const std::vector<Ort::Value> &outputs);
};

