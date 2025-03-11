#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>

#include "sam2_encoder.hpp"
#include "utils.hpp"

SAM2ImageEncoder::SAM2ImageEncoder(const std::string &path, const std::string &model_precision)
{
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    session_options.SetLogSeverityLevel(4);
    session_options.SetIntraOpNumThreads(1);
    // Optimization will take time and memory during startup
    // sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    cuda_options.device_id = 0;                                             // GPU_ID
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
    cuda_options.arena_extend_strategy = 0;
    // May cause data race in some condition
    cuda_options.do_copy_in_default_stream = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options

    session_ = Ort::Session(env_, path.c_str(), session_options);

    GetInputDetails();
    GetOutputDetails();
    model_precision_ = (model_precision == "fp16") ? CV_16F : CV_32F;
}

void SAM2ImageEncoder::EncodeImage(const std::vector<cv::Mat> &images)
{
    cv::Mat input_tensor = PrepareInput(images);
    std::vector<Ort::Value> outputs = Infer(input_tensor);
    ProcessOutput(outputs);
}

void SAM2ImageEncoder::GetInputDetails()
{
    const Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    const auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto input_shape = input_tensor_info.GetShape();

    input_height_ = input_shape[2];
    input_width_ = input_shape[3];

    auto input_name = session_.GetInputNameAllocated(0, allocator);
    input_names_.push_back(input_name.get());
}

void SAM2ImageEncoder::GetOutputDetails()
{
    for (size_t i = 0; i < session_.GetOutputCount(); ++i)
    {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(output_name.get());
    }
}


// waiting for cuda accel
cv::Mat SAM2ImageEncoder::PrepareInput(const std::vector<cv::Mat> &images)
{
    cv::Scalar mean(123.675, 116.28, 103.53); // RGB 均值
    std::vector<float> std{0.229f, 0.224f, 0.225f};  // RGB 标准差

    int num_images = images.size();
    batch_size_ = num_images;

    // mean, normalize to 0~1, to NCHW
    cv::Mat normalized_images = cv::dnn::blobFromImages(images, 1.0 / 255.0, cv::Size(input_width_, input_height_), mean, true, false, CV_32F);
    // normalize std    
    auto ptr = normalized_images.ptr<float>();
    for(int n = 0;n < num_images; ++n)
    {
        auto bias_batch = n * 3 * input_height_ * input_width_;
        for (int i = 0; i < 3; i++)
        {
            auto bias_channel = i * input_height_ * input_width_;
            for (int j = 0; j < input_height_ * input_width_; ++j)
            {
                ptr[bias_batch + bias_channel + j] /= std[i];
            }
        }
    }
    normalized_images.convertTo(normalized_images, CV_16F);
    return normalized_images;
}

std::vector<Ort::Value> SAM2ImageEncoder::Infer(const cv::Mat &input_tensor)
{
    std::vector<int64_t> input_shape = {batch_size_, 3, static_cast<int64_t>(input_height_), static_cast<int64_t>(input_width_)};
    size_t input_tensor_size = batch_size_ * 3 * input_height_ * input_width_;

    std::vector<Ort::Float16_t> input_tensor_values(input_tensor.begin<cv::float16_t>(), input_tensor.end<cv::float16_t>());
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput);


    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

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

    return session_.Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor_ort, 1,
                        output_names_cstr.data(), output_names_.size());
}

void SAM2ImageEncoder::ProcessOutput(const std::vector<Ort::Value> &outputs)
{
    feats_0_data = outputs[0].GetTensorData<Ort::Float16_t>();

    feats_1_data = outputs[1].GetTensorData<Ort::Float16_t>();

    embed_data = outputs[2].GetTensorData<Ort::Float16_t>();
}
