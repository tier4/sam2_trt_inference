#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "omp.h"

#include "sam2_decoder.hpp"

SAM2ImageDecoder::SAM2ImageDecoder(const std::string &path,
                                   const std::string &model_precision,
                                   const cv::Size &encoder_input_size,
                                   float mask_threshold)
    : encoder_input_size_(encoder_input_size), mask_threshold_(mask_threshold)
{
    try
    {
        Ort::SessionOptions session_options;
        OrtCUDAProviderOptions cuda_options;
        session_options.SetLogSeverityLevel(4);
        session_options.SetIntraOpNumThreads(1);

        cuda_options.device_id = 0;                                             // GPU_ID
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
        cuda_options.arena_extend_strategy = 0;

        // May cause data race in some condition
        cuda_options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options

        session_ = Ort::Session(env_, path.c_str(), session_options);

        GetInputDetails();
        GetOutputDetails();
        model_precision_ = (model_precision == "fp16") ? CV_16F : CV_32F;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Failed to create ONNX Runtime session: " << e.what() << std::endl;
        throw;
    }
}

void SAM2ImageDecoder::Predict(const Ort::Float16_t *image_embed, const Ort::Float16_t *high_res_feats_0, const Ort::Float16_t *high_res_feats_1,
                                const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels,
                                const cv::Size &orig_im_size, const int batch_idx)
{
    ResetVariables();
    PrepareInputs(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, orig_im_size, batch_idx);

    auto outputs = Infer();

    ProcessOutput(outputs, orig_im_size);
}

void SAM2ImageDecoder::GetInputDetails()
{
    for (size_t i = 0; i < session_.GetInputCount(); ++i)
    {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_names_.push_back(input_name.get());
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_shape = tensor_info.GetShape();
        input_shapes_.push_back(input_shape);
    }
}

void SAM2ImageDecoder::GetOutputDetails()
{
    for (size_t i = 0; i < session_.GetOutputCount(); ++i)
    {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}

void SAM2ImageDecoder::PrepareInputs(const Ort::Float16_t *image_embed, const Ort::Float16_t *high_res_feats_0, const Ort::Float16_t *high_res_feats_1,
                                     const std::vector<std::vector<cv::Point2f>> &point_coords, const std::vector<std::vector<float>> &point_labels,
                                     const cv::Size &orig_im_size, const int batch_idx)
{
    // Normalize point coordinates
    for (const auto &point : point_coords)
    {
        // 将点坐标归一化
        // 左上角坐标
        normalized_coords_.push_back(Ort::Float16_t(point[0].x / orig_im_size.width * encoder_input_size_.width));
        normalized_coords_.push_back(Ort::Float16_t(point[0].y / orig_im_size.height * encoder_input_size_.height));
        // 右下角坐标
        normalized_coords_.push_back(Ort::Float16_t(point[1].x / orig_im_size.width * encoder_input_size_.width));
        normalized_coords_.push_back(Ort::Float16_t(point[1].y / orig_im_size.height * encoder_input_size_.height));
    }
    for (const auto &label : point_labels) {
        point_labels_copy_.push_back(Ort::Float16_t(label[0]));
        point_labels_copy_.push_back(Ort::Float16_t(label[1]));
    }

    // Prepare tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput);
    int num_labels = point_labels.size();
    int num_points = point_coords[0].size();

    // image_embed
    std::vector<int64_t> image_embed_shape = input_shapes_[0];
    int64_t image_embed_size = std::accumulate(image_embed_shape.begin(), image_embed_shape.end(), 1LL, std::multiplies<int64_t>());
    image_embed_values_ = std::vector<Ort::Float16_t>(image_embed+batch_idx*image_embed_size, image_embed + (batch_idx+1)*image_embed_size);
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, image_embed_values_.data(), image_embed_values_.size(), image_embed_shape.data(), image_embed_shape.size()));

    // high_res_feats_0
    std::vector<int64_t> high_res_feats_0_shape = input_shapes_[1];
    int64_t high_res_feats_0_size = std::accumulate(high_res_feats_0_shape.begin(), high_res_feats_0_shape.end(), 1LL, std::multiplies<int64_t>());
    high_res_feats_0_values_ = std::vector<Ort::Float16_t>(high_res_feats_0+batch_idx*high_res_feats_0_size, high_res_feats_0 + (batch_idx+1)*high_res_feats_0_size);
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, high_res_feats_0_values_.data(), high_res_feats_0_values_.size(), high_res_feats_0_shape.data(), high_res_feats_0_shape.size()));

    // high_res_feats_1
    std::vector<int64_t> high_res_feats_1_shape = input_shapes_[2];
    int64_t high_res_feats_1_size = std::accumulate(high_res_feats_1_shape.begin(), high_res_feats_1_shape.end(), 1LL, std::multiplies<int64_t>());
    high_res_feats_1_values_ = std::vector<Ort::Float16_t>(high_res_feats_1+batch_idx*high_res_feats_1_size, high_res_feats_1 + (batch_idx+1)*high_res_feats_1_size);
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, high_res_feats_1_values_.data(), high_res_feats_1_values_.size(), high_res_feats_1_shape.data(), high_res_feats_1_shape.size()));

    // point_coords
    std::vector<int64_t> normalized_coords_shape = {num_labels, num_points, 2};
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, normalized_coords_.data(), normalized_coords_.size(), normalized_coords_shape.data(), normalized_coords_shape.size()));

    // point_labels
    std::vector<int64_t> point_labels_shape = {num_labels, num_points};
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, point_labels_copy_.data(), point_labels_copy_.size(), point_labels_shape.data(), point_labels_shape.size()));
    
    // mask_input
    int scaled_height = encoder_input_size_.height / scale_factor;
    int scaled_width = encoder_input_size_.width / scale_factor;
    int mask_size = num_labels * scaled_height * scaled_width;
    std::vector<int64_t> mask_input_shape = {num_labels, 1, scaled_height, scaled_width};
    mask_input_ = std::vector<Ort::Float16_t>(mask_size, Ort::Float16_t(0.0f));
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, mask_input_.data(), mask_input_.size(), mask_input_shape.data(), mask_input_shape.size()));

    // has_mask_input
    std::vector<int64_t> has_mask_input_shape = {1};
    has_mask_input_ = std::vector<Ort::Float16_t>(1, Ort::Float16_t(0.0f));
    inputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, has_mask_input_.data(), has_mask_input_.size(), has_mask_input_shape.data(), has_mask_input_shape.size()));
    
}

std::vector<Ort::Value> SAM2ImageDecoder::Infer()
{
    std::vector<const char *> input_names_cstr;
    for (const auto &name : input_names_)
    {
        input_names_cstr.push_back(name.c_str());
    }

    std::vector<const char *> output_names_cstr;
    for (const auto &name : output_names_)
    {
        output_names_cstr.push_back(name.c_str());
    }
    
    std::vector<Ort::Value> ort_outputs = session_.Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), inputs_.data(), inputs_.size(),
                        output_names_cstr.data(), output_names_.size());

    return ort_outputs;
}

void SAM2ImageDecoder::ProcessOutput(const std::vector<Ort::Value> &outputs, const cv::Size &orig_im_size)
{
    const Ort::Float16_t *mask_data = outputs[0].GetTensorData<Ort::Float16_t>();
    std::vector<int64_t> mask_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int64_t batch_size = mask_shape[0];
    const int64_t h = mask_shape[2], w = mask_shape[3];

    result_masks.resize(batch_size);

    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        const Ort::Float16_t *mask_data_i = mask_data + i * h * w;
        
        // 先创建一个临时的float矩阵
        std::vector<float> float_data(h * w);
        for(int j = 0; j < h * w; j++) {
            float_data[j] = float(mask_data_i[j]);
        }
        
        // 使用float数据创建Mat
        cv::Mat mask_i(h, w, CV_32FC1, float_data.data());
        cv::resize(mask_i, mask_i, orig_im_size, cv::INTER_LINEAR);
        // mask_i = mask_i > mask_threshold_;
        cv::threshold(mask_i, mask_i, mask_threshold_, 255, cv::THRESH_BINARY);
        mask_i.convertTo(mask_i, CV_8U); // 转换到可显示范围

        // 线程安全地写入预分配位置
        result_masks[i] = mask_i.clone();
    }
}

void SAM2ImageDecoder::ResetVariables()
{
    inputs_.clear();
    image_embed_values_.clear();
    high_res_feats_0_values_.clear();
    high_res_feats_1_values_.clear();
    normalized_coords_.clear();
    point_labels_copy_.clear();
    mask_input_.clear();
    has_mask_input_.clear();
    result_masks.clear();
}