#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include "sam2_decoder.hpp"
#include "sam2_encoder.hpp"

#include <tensorrt_common/tensorrt_common.hpp>
#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>

class SAM2Image
{
public:
    // 构造函数
    SAM2Image(const std::string &encoder_path, const std::string &decoder_path, const cv::Size encoder_input_size, const std::string &model_precision, const int decoder_batch_limit);

    // 设置输入图像
    void RunEncoder(const std::vector<cv::Mat> &images);


    // 设置一个矩形框并生成掩码
    void RunDecoder(const std::vector<std::vector<cv::Rect>> &boxes);

    // 解码掩码
    void DecodeMask(const cv::Size &orig_im_size, const int img_batch_idx, std::vector<cv::Mat> &masks_per_image, const int current_batch_size);

    // 获取生成的所有掩码
    const std::vector<std::vector<cv::Mat>> &GetMasks();

private:
    // 清除框的坐标和标签
    void ClearBoxes();

    // Encoder 对象
    std::unique_ptr<SAM2ImageEncoder> encoder_;

    // Decoder 对象
    std::unique_ptr<SAM2ImageDecoder> decoder_;

    // 解码器的路径
    std::string decoder_path_;

    // 解码器的bbox批量限制
    int decoder_batch_limit_;

    // 模型精度（fp16 或 fp32）
    std::string model_precision_;

    // Encoder 的中间特征
    CudaUniquePtrHost<float[]> high_res_feats_0_;
    CudaUniquePtrHost<float[]> high_res_feats_1_;
    CudaUniquePtrHost<float[]> image_embed_;

    // 框和掩码
    std::vector<std::vector<cv::Mat>> masks_;
    std::vector<std::vector<cv::Point2f>> box_coords_;
    std::vector<std::vector<float>> box_labels_;

    // 输入图像的原始尺寸
    std::vector<cv::Size> orig_im_size_;
};

