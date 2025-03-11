#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <numeric>

class SAM2ImageDecoder
{
public:
    // 构造函数
    SAM2ImageDecoder(const std::string &path, const std::string &model_precision, 
                     const cv::Size &encoder_input_size, float mask_threshold = 0.0);

    // 预测方法
    void Predict(const Ort::Float16_t *image_embed, const Ort::Float16_t *high_res_feats_0, const Ort::Float16_t *high_res_feats_1,
                 const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels,
                 const cv::Size &orig_im_size, const int batch_idx);

    // 结果掩码
    std::vector<cv::Mat> result_masks;

private:
    // ort信息
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "SAM2ImageDecoder"};
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    
    // I/O张量信息
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    cv::Size encoder_input_size_;

    // 其他信息
    int model_precision_;
    float mask_threshold_;
    int scale_factor = 4;
    

    // 输入张量数据
    std::vector<Ort::Value> inputs_;
    std::vector<Ort::Float16_t> image_embed_values_;
    std::vector<Ort::Float16_t> high_res_feats_0_values_;
    std::vector<Ort::Float16_t> high_res_feats_1_values_;
    std::vector<Ort::Float16_t> normalized_coords_;
    std::vector<Ort::Float16_t> point_labels_copy_;
    std::vector<Ort::Float16_t> mask_input_;
    std::vector<Ort::Float16_t> has_mask_input_;

    // 获取输入张量信息
    void GetInputDetails();

    // 获取输出张量信息
    void GetOutputDetails();

    // 准备输入数据
    void PrepareInputs(const Ort::Float16_t *image_embed, const Ort::Float16_t *high_res_feats_0, const Ort::Float16_t *high_res_feats_1, const std::vector<std::vector<cv::Point2f>> &point_coords, 
                                          const std::vector<std::vector<float>> &point_labels, const cv::Size &orig_im_size, const int batch_idx);

    // 推理过程
    std::vector<Ort::Value> Infer();

    // 处理推理结果
    void ProcessOutput(const std::vector<Ort::Value> &outputs, const cv::Size &orig_im_size);

    // 重置所有变量
    void ResetVariables();
};
