/**
 * @file sam2_decoder.hpp
 * @brief SAM2 decoder implementation using TensorRT
 * 
 * Copyright (c) 2024 TIERIV
 * Author: Hunter Cheng (haoxuan.cheng@tier4.jp)
 * Created: 2025.4
 */

#pragma once

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
    // Constructor
    SAM2ImageDecoder(const std::string &onnx_path, 
                    const std::string &engine_precision,
                    const tensorrt_common::BatchConfig &batch_config,
                    const size_t max_workspace_size,
                    const tensorrt_common::BuildConfig build_config,
                    const cv::Size &encoder_input_size, 
                    const std::vector<int> &encoder_output_sizes,
                    float mask_threshold = 0.0);
    ~SAM2ImageDecoder();

    // Prediction method
    void Predict(CudaUniquePtrHost<float[]> &image_embed, CudaUniquePtrHost<float[]> &high_res_feats_0, CudaUniquePtrHost<float[]> &high_res_feats_1,
                 const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels,
                 const cv::Size &orig_im_size, const int batch_idx, const int current_batch_size);

    // Result masks
    std::vector<cv::Mat> result_masks;
    std::unique_ptr<tensorrt_common::TrtCommon> trt_decoder_;

    // CPU input data
    CudaUniquePtrHost<float[]> image_embed_data;
    CudaUniquePtrHost<float[]> high_res_feats_0_data;
    CudaUniquePtrHost<float[]> high_res_feats_1_data;
    CudaUniquePtrHost<float[]> normalized_coords_data;
    CudaUniquePtrHost<float[]> point_labels_data;
    CudaUniquePtrHost<float[]> mask_input_data;
    CudaUniquePtrHost<float[]> has_mask_input_data;
    // CPU output data
    CudaUniquePtrHost<float[]> output_mask_data;
    CudaUniquePtrHost<float[]> output_confidence_data;
    

private:
    // Allocate GPU memory
    void allocateGpuMemory();

    // Calculate memory size
    void CalculateMemorySize(const int decoder_batch_limit, const int image_embed_size, const int high_res_feats_0_size, const int high_res_feats_1_size);
    
    // I/O tensor information
    std::vector<std::vector<int64_t>> input_output_shapes_;
    cv::Size encoder_input_size_;

    // Other information
    int model_precision_;
    float mask_threshold_;
    int scale_factor = 4;
    

    // GPU input data
    CudaUniquePtr<float[]> image_embed_data_d_;
    CudaUniquePtr<float[]> high_res_feats_0_data_d_;
    CudaUniquePtr<float[]> high_res_feats_1_data_d_;
    CudaUniquePtr<float[]> normalized_coords_data_d_;
    CudaUniquePtr<float[]> point_labels_data_d_;
    CudaUniquePtr<float[]> mask_input_data_d_;
    CudaUniquePtr<float[]> has_mask_input_data_d_;
    // GPU output data
    CudaUniquePtr<float[]> output_mask_data_d_;
    CudaUniquePtr<float[]> output_confidence_data_d_;

    // Get input tensor information
    void GetInputOutputDetails();

    // Prepare input data
    void PrepareInputs(const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels, 
                       const cv::Size &orig_im_size);

    // Inference process
    bool Infer(CudaUniquePtrHost<float[]> &image_embed, CudaUniquePtrHost<float[]> &high_res_feats_0, CudaUniquePtrHost<float[]> &high_res_feats_1,
               const int batch_idx);

    // Process inference results
    void ProcessOutput(const cv::Size &orig_im_size, const int current_batch_size);

    // Reset all variables
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
