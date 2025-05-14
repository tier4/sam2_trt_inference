/**
 * @file sam2_decoder.cpp
 * @brief Implementation of SAM2 decoder using TensorRT
 *
 * Copyright (c) 2024 TIERIV
 * Author: Hunter Cheng (haoxuan.cheng@tier4.jp)
 * Created: 2025.4
 */

#include "sam2_decoder.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "omp.h"

SAM2ImageDecoder::SAM2ImageDecoder(const std::string& onnx_path,
                                   const std::string& engine_precision,
                                   const tensorrt_common::BatchConfig& batch_config,
                                   const size_t max_workspace_size,
                                   const tensorrt_common::BuildConfig build_config,
                                   const cv::Size& encoder_input_size,
                                   const std::vector<int>& encoder_output_sizes,
                                   float mask_threshold)
    : encoder_input_size_(encoder_input_size),
      mask_threshold_(mask_threshold)
{
    trt_decoder_ = std::make_unique<tensorrt_common::TrtCommon>(
        onnx_path, engine_precision, nullptr, batch_config, max_workspace_size, build_config);

    trt_decoder_->setup();

    if (!trt_decoder_->isInitialized())
    {
        throw std::runtime_error("Failed to initialize TRT decoder");
        return;
    }

    CalculateMemorySize(
        batch_config[2], encoder_output_sizes[0], encoder_output_sizes[1], encoder_output_sizes[2]);
    AllocateGPUMemory();
    GetInputOutputDetails();
}

SAM2ImageDecoder::~SAM2ImageDecoder()
{
}

void SAM2ImageDecoder::AllocateGPUMemory()
{
    // CPU part
    normalized_coords_data =
        cuda_utils::make_unique_host<float[]>(normalized_coords_size_, cudaHostAllocPortable);
    point_labels_data =
        cuda_utils::make_unique_host<float[]>(point_labels_size_, cudaHostAllocPortable);
    mask_input_data =
        cuda_utils::make_unique_host<float[]>(mask_input_size_, cudaHostAllocPortable);
    has_mask_input_data =
        cuda_utils::make_unique_host<float[]>(has_mask_input_size_, cudaHostAllocPortable);
    output_mask_data =
        cuda_utils::make_unique_host<float[]>(output_mask_size_, cudaHostAllocPortable);
    output_confidence_data =
        cuda_utils::make_unique_host<float[]>(output_confidence_size_, cudaHostAllocPortable);

    // GPU part
    image_embed_data_d_ = cuda_utils::make_unique<float[]>(image_embed_size_);
    high_res_feats_0_data_d_ = cuda_utils::make_unique<float[]>(high_res_feats_0_size_);
    high_res_feats_1_data_d_ = cuda_utils::make_unique<float[]>(high_res_feats_1_size_);
    normalized_coords_data_d_ = cuda_utils::make_unique<float[]>(normalized_coords_size_);
    point_labels_data_d_ = cuda_utils::make_unique<float[]>(point_labels_size_);
    mask_input_data_d_ = cuda_utils::make_unique<float[]>(mask_input_size_);
    has_mask_input_data_d_ = cuda_utils::make_unique<float[]>(has_mask_input_size_);
    output_mask_data_d_ = cuda_utils::make_unique<float[]>(output_mask_size_);
    output_confidence_data_d_ = cuda_utils::make_unique<float[]>(output_confidence_size_);
}

void SAM2ImageDecoder::Predict(CudaUniquePtrHost<float[]>& image_embed,
                               CudaUniquePtrHost<float[]>& high_res_feats_0,
                               CudaUniquePtrHost<float[]>& high_res_feats_1,
                               const std::vector<std::vector<cv::Point2f>>& point_coords,
                               const std::vector<std::vector<float>>& point_labels,
                               const cv::Size& orig_im_size,
                               const int batch_idx,
                               const int current_batch_size)
{
    ResetVariables();

    PrepareInputs(point_coords, point_labels, orig_im_size);

    bool success = Infer(image_embed, high_res_feats_0, high_res_feats_1, batch_idx);
    if (!success)
    {
        throw std::runtime_error("Failed to execute inference");
        return;
    }

    ProcessOutput(orig_im_size, current_batch_size);
}

void SAM2ImageDecoder::GetInputOutputDetails()
{
    for (int i = 0; i < trt_decoder_->getNbBindings(); i++)
    {
        auto dims = trt_decoder_->getBindingDimensions(i);
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; j++)
        {
            shape.push_back(dims.d[j]);
        }
        input_output_shapes_.push_back(shape);
    }
}

void SAM2ImageDecoder::CalculateMemorySize(const int decoder_batch_limit,
                                           const int image_embed_size,
                                           const int high_res_feats_0_size,
                                           const int high_res_feats_1_size)
{
    // Output from encoder
    image_embed_size_ = image_embed_size;
    high_res_feats_0_size_ = high_res_feats_0_size;
    high_res_feats_1_size_ = high_res_feats_1_size;

    // Bounding box prompt
    std::vector<int64_t> normalized_coords_shape = {decoder_batch_limit, 2, 2};
    std::vector<int64_t> point_labels_shape = {decoder_batch_limit, 2};
    normalized_coords_size_ = std::accumulate(
        normalized_coords_shape.begin(), normalized_coords_shape.end(), 1, std::multiplies<int>());
    point_labels_size_ = std::accumulate(
        point_labels_shape.begin(), point_labels_shape.end(), 1, std::multiplies<int>());

    // Mask input
    int scaled_height = encoder_input_size_.height / scale_factor;
    int scaled_width = encoder_input_size_.width / scale_factor;
    std::vector<int64_t> mask_input_shape = {decoder_batch_limit, 1, scaled_height, scaled_width};
    mask_input_size_ = std::accumulate(
        mask_input_shape.begin(), mask_input_shape.end(), 1, std::multiplies<int>());

    // Has mask input
    std::vector<int64_t> has_mask_input_shape = {1};
    has_mask_input_size_ = std::accumulate(
        has_mask_input_shape.begin(), has_mask_input_shape.end(), 1, std::multiplies<int>());

    // Output mask
    std::vector<int64_t> output_mask_shape = {decoder_batch_limit, 1, scaled_height, scaled_width};
    output_mask_size_ = std::accumulate(
        output_mask_shape.begin(), output_mask_shape.end(), 1, std::multiplies<int>());

    // Output confidence
    std::vector<int64_t> output_confidence_shape = {decoder_batch_limit};
    output_confidence_size_ = std::accumulate(
        output_confidence_shape.begin(), output_confidence_shape.end(), 1, std::multiplies<int>());
}

void SAM2ImageDecoder::PrepareInputs(const std::vector<std::vector<cv::Point2f>>& point_coords,
                                     const std::vector<std::vector<float>>& point_labels,
                                     const cv::Size& orig_im_size)
{
    // Normalize point coordinates
    int coords_idx = 0;
    for (int i = 0; i < static_cast<int>(point_coords.size()); i++)
    {
        // Normalize point coordinates to encoder input size
        for (int j = 0; j < static_cast<int>(point_coords[i].size()); j++)
        {
            normalized_coords_data[coords_idx++] =
                point_coords[i][j].x / orig_im_size.width * encoder_input_size_.width;
            normalized_coords_data[coords_idx++] =
                point_coords[i][j].y / orig_im_size.height * encoder_input_size_.height;
        }
    }

    int labels_idx = 0;
    for (int i = 0; i < static_cast<int>(point_labels.size()); i++)
    {
        for (int j = 0; j < static_cast<int>(point_labels[i].size()); j++)
        {
            point_labels_data[labels_idx++] = point_labels[i][j];
        }
    }

    // Initialize mask input
    for (int i = 0; i < mask_input_size_; i++)
    {
        mask_input_data[i] = 0.0f;
    }

    // Initialize has mask input
    has_mask_input_data[0] = 0.0f;

    // Set dynamic input dimensions
    // Current batch size
    int current_batch_size = point_coords.size();
    // Normalized coordinates
    std::vector<int64_t> normalized_coords_shape = {current_batch_size, 2, 2};
    nvinfer1::Dims normalized_coords_dims;
    normalized_coords_dims.nbDims = 3;
    normalized_coords_dims.d[0] = normalized_coords_shape[0];
    normalized_coords_dims.d[1] = normalized_coords_shape[1];
    normalized_coords_dims.d[2] = normalized_coords_shape[2];
    trt_decoder_->setBindingDimensions(3, normalized_coords_dims);

    // Point labels
    nvinfer1::Dims point_labels_dims;
    point_labels_dims.nbDims = 2;
    point_labels_dims.d[0] = current_batch_size;
    point_labels_dims.d[1] = 2;
    trt_decoder_->setBindingDimensions(4, point_labels_dims);

    // Mask input
    nvinfer1::Dims mask_input_dims;
    mask_input_dims.nbDims = 4;
    mask_input_dims.d[0] = current_batch_size;
    mask_input_dims.d[1] = 1;
    mask_input_dims.d[2] = encoder_input_size_.height / scale_factor;
    mask_input_dims.d[3] = encoder_input_size_.width / scale_factor;
    trt_decoder_->setBindingDimensions(5, mask_input_dims);
}

bool SAM2ImageDecoder::Infer(CudaUniquePtrHost<float[]>& image_embed,
                             CudaUniquePtrHost<float[]>& high_res_feats_0,
                             CudaUniquePtrHost<float[]>& high_res_feats_1,
                             const int batch_idx)
{
    // Copy fixed shape inputs
    CHECK_CUDA_ERROR(cudaMemcpyAsync(image_embed_data_d_.get(),
                                     image_embed.get() + batch_idx * image_embed_size_,
                                     image_embed_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(high_res_feats_0_data_d_.get(),
                                     high_res_feats_0.get() + batch_idx * high_res_feats_0_size_,
                                     high_res_feats_0_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(high_res_feats_1_data_d_.get(),
                                     high_res_feats_1.get() + batch_idx * high_res_feats_1_size_,
                                     high_res_feats_1_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));

    // Copy dynamic shape inputs
    CHECK_CUDA_ERROR(cudaMemcpyAsync(normalized_coords_data_d_.get(),
                                     normalized_coords_data.get(),
                                     normalized_coords_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(point_labels_data_d_.get(),
                                     point_labels_data.get(),
                                     point_labels_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(mask_input_data_d_.get(),
                                     mask_input_data.get(),
                                     mask_input_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(has_mask_input_data_d_.get(),
                                     has_mask_input_data.get(),
                                     has_mask_input_size_ * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     *stream_));

    // Prepare GPU buffers
    std::vector<void*> buffers = {image_embed_data_d_.get(),
                                  high_res_feats_0_data_d_.get(),
                                  high_res_feats_1_data_d_.get(),
                                  normalized_coords_data_d_.get(),
                                  point_labels_data_d_.get(),
                                  mask_input_data_d_.get(),
                                  has_mask_input_data_d_.get(),
                                  output_mask_data_d_.get(),
                                  output_confidence_data_d_.get()};

    // Execute inference
    bool success = trt_decoder_->enqueueV2(buffers.data(), *stream_, nullptr);
    if (!success)
    {
        throw std::runtime_error("Failed to execute inference");
        return false;
    }

    // Copy output
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_mask_data.get(),
                                     output_mask_data_d_.get(),
                                     output_mask_size_ * sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_confidence_data.get(),
                                     output_confidence_data_d_.get(),
                                     output_confidence_size_ * sizeof(float),
                                     cudaMemcpyDeviceToHost,
                                     *stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(*stream_));
    return true;
}

void SAM2ImageDecoder::ProcessOutput(const cv::Size& orig_im_size, const int current_batch_size)
{
    const float* mask_data = output_mask_data.get();
    auto mask_dims = trt_decoder_->getBindingDimensions(7);
    std::vector<int64_t> mask_shape = {
        current_batch_size, mask_dims.d[1], mask_dims.d[2], mask_dims.d[3]};
    const int64_t h = mask_shape[2], w = mask_shape[3];

    result_masks.resize(current_batch_size);

#pragma omp parallel for
    for (int i = 0; i < current_batch_size; i++)
    {
        // Create Mat directly from mask_data_i to avoid unnecessary data copy
        cv::Mat mask_i(
            h, w, CV_32FC1, const_cast<void*>(static_cast<const void*>(mask_data + i * h * w)));

        // Perform resize and threshold operations in one step
        cv::Mat resized_mask;
        cv::resize(mask_i, resized_mask, orig_im_size, 0, 0, cv::INTER_LINEAR);

        // Convert to 8-bit and binarize
        cv::Mat binary_mask;
        resized_mask = resized_mask > mask_threshold_;
        resized_mask.convertTo(binary_mask, CV_8U, 255);

        result_masks[i] = binary_mask;
    }
}

void SAM2ImageDecoder::ResetVariables()
{
    result_masks.clear();
}