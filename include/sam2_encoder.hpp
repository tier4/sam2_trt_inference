#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include <tensorrt_common/tensorrt_common.hpp>
#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>

using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

class SAM2ImageEncoder
{
public:
    // 构造函数
    SAM2ImageEncoder(const std::string &onnx_path, 
                    const std::string &engine_precision,
                    const tensorrt_common::BatchConfig &batch_config,
                    const size_t max_workspace_size,
                    const tensorrt_common::BuildConfig build_config);

    ~SAM2ImageEncoder();

    // 对图像进行编码
    void EncodeImage(const std::vector<cv::Mat> &images);

    // 编码后的高分辨率特征
    CudaUniquePtrHost<float[]> feats_0_data;
    CudaUniquePtrHost<float[]> feats_1_data;
    CudaUniquePtrHost<float[]> embed_data;

    // 输入的宽高
    int input_height_;
    int input_width_;

    // 模型的精度（fp16 或 fp32）
    std::string encoder_precision_;

    // 批处理大小
    int batch_size_;
    int feats_0_size_;
    int feats_1_size_;
    int embed_size_;

private:
    // 分配 GPU 内存
    void allocateGpuMemory();

    // 从 ONNX 模型获取输入的详细信息
    void GetInputDetails();

    // 从 ONNX 模型获取输出的详细信息
    void GetOutputDetails();

    // 准备输入张量
    cv::Mat PrepareInput(const std::vector<cv::Mat> &images);

    // 执行推理
    bool Infer(const cv::Mat &input_tensor);

    // 处理推理输出
    void ProcessOutput();

    std::unique_ptr<tensorrt_common::TrtCommon> trt_encoder_;

    CudaUniquePtr<float[]> input_d_;
    CudaUniquePtr<float[]> feats_0_data_d_;
    CudaUniquePtr<float[]> feats_1_data_d_;
    CudaUniquePtr<float[]> embed_data_d_;

    StreamUniquePtr stream_{makeCudaStream()};
};

