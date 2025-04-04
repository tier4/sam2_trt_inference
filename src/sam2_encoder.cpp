#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "sam2_encoder.hpp"

SAM2ImageEncoder::SAM2ImageEncoder(
    const std::string &onnx_path, const std::string &engine_precision,
    const tensorrt_common::BatchConfig &batch_config,
    const size_t max_workspace_size,
    const tensorrt_common::BuildConfig build_config)
    : encoder_precision_(engine_precision) {
  trt_encoder_ = std::make_unique<tensorrt_common::TrtCommon>(
      onnx_path, engine_precision, nullptr, batch_config, max_workspace_size,
      build_config);

  trt_encoder_->setup();

  if (!trt_encoder_->isInitialized()) {
    throw std::runtime_error("Failed to initialize TRT encoder");
    return;
  }

  GetInputDetails();
  GetOutputDetails();

  allocateGpuMemory();
}

SAM2ImageEncoder::~SAM2ImageEncoder() {}

void SAM2ImageEncoder::allocateGpuMemory() {
  const auto input_dims = trt_encoder_->getBindingDimensions(0);
  const auto embed_dims = trt_encoder_->getBindingDimensions(1);
  const auto feats_1_dims = trt_encoder_->getBindingDimensions(2);
  const auto feats_0_dims = trt_encoder_->getBindingDimensions(3);

  const auto input_size =
      std::accumulate(input_dims.d + 1, input_dims.d + input_dims.nbDims, 1,
                      std::multiplies<int>());
  feats_0_size_ =
      std::accumulate(feats_0_dims.d + 1, feats_0_dims.d + feats_0_dims.nbDims,
                      1, std::multiplies<int>());
  feats_1_size_ =
      std::accumulate(feats_1_dims.d + 1, feats_1_dims.d + feats_1_dims.nbDims,
                      1, std::multiplies<int>());
  embed_size_ =
      std::accumulate(embed_dims.d + 1, embed_dims.d + embed_dims.nbDims, 1,
                      std::multiplies<int>());

  // CPU part
  feats_0_data = cuda_utils::make_unique_host<float[]>(feats_0_size_,
                                                       cudaHostAllocPortable);
  feats_1_data = cuda_utils::make_unique_host<float[]>(feats_1_size_,
                                                       cudaHostAllocPortable);
  embed_data =
      cuda_utils::make_unique_host<float[]>(embed_size_, cudaHostAllocPortable);

  // GPU part
  input_d_ = cuda_utils::make_unique<float[]>(input_size);
  feats_0_data_d_ = cuda_utils::make_unique<float[]>(feats_0_size_);
  feats_1_data_d_ = cuda_utils::make_unique<float[]>(feats_1_size_);
  embed_data_d_ = cuda_utils::make_unique<float[]>(embed_size_);
}

void SAM2ImageEncoder::EncodeImage(const std::vector<cv::Mat> &images) {
  cv::Mat input_tensor = PrepareInput(images);
  bool success = Infer(input_tensor);
  if (!success) {
    throw std::runtime_error("Failed to encode image");
    return;
  }
  ProcessOutput();
}

void SAM2ImageEncoder::GetInputDetails() {
  const auto input_dims = trt_encoder_->getBindingDimensions(0);
  batch_size_ = input_dims.d[0];
  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];
}

void SAM2ImageEncoder::GetOutputDetails() {}

// waiting for cuda accel
cv::Mat SAM2ImageEncoder::PrepareInput(const std::vector<cv::Mat> &images) {
  cv::Scalar mean(123.675, 116.28, 103.53);       // RGB 均值
  std::vector<float> std{0.229f, 0.224f, 0.225f}; // RGB 标准差

  int num_images = images.size();
  assert(num_images <= batch_size_);

  // mean, normalize to 0~1, to NCHW
  cv::Mat normalized_images = cv::dnn::blobFromImages(
      images, 1.0 / 255.0, cv::Size(input_width_, input_height_), mean, true,
      false, CV_32F);
  // normalize std
  auto ptr = normalized_images.ptr<float>();
  for (int n = 0; n < num_images; ++n) {
    auto bias_batch = n * 3 * input_height_ * input_width_;
    for (int i = 0; i < 3; i++) {
      auto bias_channel = i * input_height_ * input_width_;
      for (int j = 0; j < input_height_ * input_width_; ++j) {
        ptr[bias_batch + bias_channel + j] /= std[i];
      }
    }
  }
  // normalized_images.convertTo(normalized_images, CV_16F);
  return normalized_images;
}

bool SAM2ImageEncoder::Infer(const cv::Mat &input_tensor) {
  // If the data is continuous, we can use it directly. Otherwise, we need to
  // clone it for contiguous memory.
  auto input_tensor_cloned =
      input_tensor.isContinuous()
          ? input_tensor.reshape(1, input_tensor.total())
          : input_tensor.reshape(1, input_tensor.total()).clone();
  // copy input to GPU
  CHECK_CUDA_ERROR(cudaMemcpyAsync(input_d_.get(),
                                   input_tensor_cloned.ptr<float>(),
                                   input_tensor_cloned.total() * sizeof(float),
                                   cudaMemcpyHostToDevice, *stream_));

  // prepare GPU buffers
  std::vector<void *> buffers = {
      input_d_.get(),
      embed_data_d_.get(),
      feats_1_data_d_.get(),
      feats_0_data_d_.get(),
  };

  // execute inference
  bool success = trt_encoder_->enqueueV2(buffers.data(), *stream_, nullptr);
  if (!success) {
    throw std::runtime_error("Failed to execute inference");
    return false;
  }

  // copy output to CPU
  CHECK_CUDA_ERROR(cudaMemcpyAsync(feats_0_data.get(), feats_0_data_d_.get(),
                                   feats_0_size_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, *stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(feats_1_data.get(), feats_1_data_d_.get(),
                                   feats_1_size_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, *stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(embed_data.get(), embed_data_d_.get(),
                                   embed_size_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, *stream_));

  // synchronize
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*stream_));

  return true;
}

void SAM2ImageEncoder::ProcessOutput() {}
