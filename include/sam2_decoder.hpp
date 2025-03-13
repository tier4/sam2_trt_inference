#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <numeric>

#include <tensorrt_common/tensorrt_common.hpp>
#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>

using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

class SAM2ImageDecoder
{
public:
    // 构造函数
    SAM2ImageDecoder(const std::string &onnx_path, 
                    const std::string &engine_precision,
                    const tensorrt_common::BatchConfig &batch_config,
                    const size_t max_workspace_size,
                    const tensorrt_common::BuildConfig build_config,
                    const cv::Size &encoder_input_size, 
                    float mask_threshold = 0.0);
    ~SAM2ImageDecoder();

    // 预测方法
    void Predict(CudaUniquePtrHost<float[]> &image_embed, CudaUniquePtrHost<float[]> &high_res_feats_0, CudaUniquePtrHost<float[]> &high_res_feats_1,
                 const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels,
                 const cv::Size &orig_im_size, const int batch_idx, const int image_embed_size, const int high_res_feats_0_size, const int high_res_feats_1_size);

    // 结果掩码
    std::vector<cv::Mat> result_masks;
    std::unique_ptr<tensorrt_common::TrtCommon> trt_decoder_;

    // 输入数据CPU
    CudaUniquePtrHost<float[]> image_embed_data;
    CudaUniquePtrHost<float[]> high_res_feats_0_data;
    CudaUniquePtrHost<float[]> high_res_feats_1_data;
    CudaUniquePtrHost<float[]> normalized_coords_data;
    CudaUniquePtrHost<float[]> point_labels_data;
    CudaUniquePtrHost<float[]> mask_input_data;
    CudaUniquePtrHost<float[]> has_mask_input_data;
    // 输出数据CPU
    CudaUniquePtrHost<float[]> output_mask_data;
    CudaUniquePtrHost<float[]> output_confidence_data;
    

private:
    // 分配 GPU 内存
    void allocateGpuMemory();
    
    // I/O张量信息
    std::vector<std::vector<int64_t>> input_output_shapes_;
    cv::Size encoder_input_size_;

    // 其他信息
    int model_precision_;
    float mask_threshold_;
    int scale_factor = 4;
    

    // 输入数据GPU
    CudaUniquePtr<float[]> image_embed_data_d_;
    CudaUniquePtr<float[]> high_res_feats_0_data_d_;
    CudaUniquePtr<float[]> high_res_feats_1_data_d_;
    CudaUniquePtr<float[]> normalized_coords_data_d_;
    CudaUniquePtr<float[]> point_labels_data_d_;
    CudaUniquePtr<float[]> mask_input_data_d_;
    CudaUniquePtr<float[]> has_mask_input_data_d_;
    // 输出数据GPU
    CudaUniquePtr<float[]> output_mask_data_d_;
    CudaUniquePtr<float[]> output_confidence_data_d_;

    // 获取输入张量信息
    void GetInputOutputDetails();

    // 准备输入数据
    void PrepareInputs(const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels, 
                       const cv::Size &orig_im_size, const int batch_idx, const int image_embed_size, const int high_res_feats_0_size, const int high_res_feats_1_size);

    // 推理过程
    bool Infer(CudaUniquePtrHost<float[]> &image_embed, CudaUniquePtrHost<float[]> &high_res_feats_0, CudaUniquePtrHost<float[]> &high_res_feats_1,
               const int batch_idx);

    // 处理推理结果
    void ProcessOutput(const cv::Size &orig_im_size);

    // 重置所有变量
    void ResetVariables();

    StreamUniquePtr stream_{makeCudaStream()};

    int image_embed_size_;
    int high_res_feats_0_size_;
    int high_res_feats_1_size_;
    int normalized_coords_size_;
    int point_labels_size_;
    int mask_input_size_;
    int has_mask_input_size_;
    int output_mask_size_;
    int output_confidence_size_;
};
