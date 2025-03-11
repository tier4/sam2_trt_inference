#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>

#include "sam2_image_inference.hpp"
#include "utils.hpp"

SAM2Image::SAM2Image(const std::string &encoder_path, const std::string &decoder_path, const cv::Size encoder_input_size, const std::string &model_precision, const int decoder_batch_limit)
    : encoder_(encoder_path, model_precision), decoder_(std::make_unique<SAM2ImageDecoder>(decoder_path, model_precision, encoder_input_size)), decoder_batch_limit_(decoder_batch_limit), model_precision_(model_precision) {}

void SAM2Image::RunEncoder(const std::vector<cv::Mat> &images)
{
    // 清除所有的变量
    masks_.clear();
    box_coords_.clear();
    box_labels_.clear();
    orig_im_size_.clear();

    // 跑encoder得到结果
    encoder_.EncodeImage(images);
    high_res_feats_0_ = encoder_.feats_0_data;
    high_res_feats_1_ = encoder_.feats_1_data;
    image_embed_ = encoder_.embed_data;

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
    decoder_->Predict(image_embed_, high_res_feats_0_, high_res_feats_1_, box_coords_, box_labels_, orig_im_size, img_batch_idx);
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