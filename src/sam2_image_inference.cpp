#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>

#include "sam2_image_inference.hpp"
#include "utils.hpp"

SAM2Image::SAM2Image(const std::string &encoder_path, const std::string &decoder_path, const cv::Size encoder_input_size, const std::string &model_precision, const int decoder_batch_limit)
    :decoder_batch_limit_(decoder_batch_limit), model_precision_(model_precision) 
{
    // make config
    tensorrt_common::BatchConfig batch_config_encoder = {1, 1, 1};
    tensorrt_common::BatchConfig batch_config_decoder = {1, decoder_batch_limit/2, decoder_batch_limit};
    tensorrt_common::BuildConfig build_config_encoder("Entropy", -1, false, false, false, 0.0, false, {});
    tensorrt_common::BuildConfig build_config_decoder("Entropy", -1, false, false, false, 0.0, false, {});
    const size_t max_workspace_size = 1 << 30;
    // genrate encoder and decoder
    encoder_ = std::make_unique<SAM2ImageEncoder>(encoder_path, model_precision, batch_config_encoder, max_workspace_size, build_config_encoder);
    decoder_ = std::make_unique<SAM2ImageDecoder>(decoder_path, model_precision, batch_config_decoder, max_workspace_size, build_config_decoder, encoder_input_size);
}

void SAM2Image::RunEncoder(const std::vector<cv::Mat> &images)
{
    // 清除所有的变量
    masks_.clear();
    orig_im_size_.clear();

    // 跑encoder得到结果
    encoder_->EncodeImage(images);

    for (const auto &image : images) {
        orig_im_size_.push_back(image.size());
    }
}

void SAM2Image::RunDecoder(const std::vector<std::vector<cv::Rect>> &boxes)
{
    assert(boxes.size() == orig_im_size_.size());
    for (size_t i = 0; i < boxes.size(); i++) {
        auto boxes_per_image = boxes[i];
        std::vector<cv::Mat> masks_per_image;

        for(int z = 0; z < int(boxes_per_image.size()); z += decoder_batch_limit_) {
            int current_batch_size = std::min(decoder_batch_limit_, int(boxes_per_image.size()) - z);
            ClearBoxes();
            // 预分配局部存储空间
            std::vector<std::vector<cv::Point2f>> local_box_coords(current_batch_size);
            std::vector<std::vector<float>> local_box_labels(current_batch_size);

            // 并行生成每个框的信息
            #pragma omp parallel for
            for (int j = 0; j < current_batch_size; j++) {
                const auto &box = boxes_per_image[z + j];
                // 计算框的两个角点
                std::vector<cv::Point2f> coords = { cv::Point2f(box.x, box.y),
                                                cv::Point2f(box.x + box.width, box.y + box.height) };
                // bbox左上角和右下角的标签数据
                std::vector<float> labels = {2, 3};

                local_box_coords[j] = coords;
                local_box_labels[j] = labels;
            }

            // 将局部结果合并到成员变量中
            box_coords_ = std::move(local_box_coords);
            box_labels_ = std::move(local_box_labels);

            DecodeMask(orig_im_size_[i], i, masks_per_image);
        }
        masks_.push_back(masks_per_image);
    }
}

void SAM2Image::DecodeMask(const cv::Size &orig_im_size, const int img_batch_idx, std::vector<cv::Mat> &masks_per_image)
{
    decoder_->Predict(encoder_->embed_data, encoder_->feats_0_data, encoder_->feats_1_data, 
                    box_coords_, box_labels_, orig_im_size, img_batch_idx, 
                    encoder_->embed_size_, encoder_->feats_0_size_, encoder_->feats_1_size_);
    auto masks_per_image_per_decoder_batch = decoder_->result_masks;
    masks_per_image.insert(masks_per_image.end(), masks_per_image_per_decoder_batch.begin(), masks_per_image_per_decoder_batch.end());
}

const std::vector<std::vector<cv::Mat>> &SAM2Image::GetMasks()
{
    return masks_;
}

void SAM2Image::ClearBoxes()
{
    box_coords_.clear();
    box_labels_.clear();
}