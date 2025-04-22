/**
 * @file sam2_image_inference.hpp
 * @brief SAM2 image inference pipeline implementation
 *
 * Copyright (c) 2024 TIERIV
 * Author: Hunter Cheng (haoxuan.cheng@tier4.jp)
 * Created: 2025.4
 */

#pragma once

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <vector>

#include "sam2_decoder.hpp"
#include "sam2_encoder.hpp"

class SAM2Image
{
   public:
    // Constructor
    SAM2Image(const std::string& encoder_path,
              const std::string& decoder_path,
              const cv::Size encoder_input_size,
              const std::string& model_precision,
              const int decoder_batch_limit);

    // Set input images
    void RunEncoder(const std::vector<cv::Mat>& images);

    // Set bounding boxes and generate masks
    void RunDecoder(const std::vector<std::vector<cv::Rect>>& boxes);

    // Decode masks
    void DecodeMask(const cv::Size& orig_im_size,
                    const int img_batch_idx,
                    std::vector<cv::Mat>& masks_per_image,
                    const int current_batch_size);

    // Get all generated masks
    const std::vector<std::vector<cv::Mat>>& GetMasks();

   private:
    // Clear box coordinates and labels
    void ClearBoxes();

    // Encoder object
    std::unique_ptr<SAM2ImageEncoder> encoder_;

    // Decoder object
    std::unique_ptr<SAM2ImageDecoder> decoder_;

    // Decoder path
    std::string decoder_path_;

    // Decoder batch size limit
    int decoder_batch_limit_;

    // Model precision (fp16 or fp32)
    std::string model_precision_;

    // Encoder intermediate features
    CudaUniquePtrHost<float[]> high_res_feats_0_;
    CudaUniquePtrHost<float[]> high_res_feats_1_;
    CudaUniquePtrHost<float[]> image_embed_;

    // Boxes and masks
    std::vector<std::vector<cv::Mat>> masks_;
    std::vector<std::vector<cv::Point2f>> box_coords_;
    std::vector<std::vector<float>> box_labels_;

    // Original input image dimensions
    std::vector<cv::Size> orig_im_size_;
};
