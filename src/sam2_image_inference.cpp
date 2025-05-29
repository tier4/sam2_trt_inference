// Copyright 2025 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#include "sam2_image_inference.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "utils.hpp"

SAM2Image::SAM2Image(const std::string& encoder_path,
                     const std::string& decoder_path,
                     const cv::Size encoder_input_size,
                     const std::string& model_precision,
                     const int decoder_batch_limit)
    : decoder_batch_limit_(decoder_batch_limit),
      model_precision_(model_precision)
{
    // Create configuration
    tensorrt_common::BatchConfig batch_config_encoder = {1, 1, 1};
    tensorrt_common::BatchConfig batch_config_decoder = {
        1, decoder_batch_limit / 2, decoder_batch_limit};
    tensorrt_common::BuildConfig build_config_encoder(
        "Entropy", -1, false, false, false, 0.0, false, {});
    tensorrt_common::BuildConfig build_config_decoder(
        "Entropy", -1, false, false, false, 0.0, false, {});
    const size_t max_workspace_size = 4ULL << 30;

    // Initialize encoder and decoder
    encoder_ = std::make_unique<SAM2ImageEncoder>(encoder_path,
                                                  model_precision,
                                                  batch_config_encoder,
                                                  max_workspace_size,
                                                  build_config_encoder);
    std::vector<int> encoder_output_sizes = {
        encoder_->embed_size_, encoder_->feats_0_size_, encoder_->feats_1_size_};
    decoder_ = std::make_unique<SAM2ImageDecoder>(decoder_path,
                                                  model_precision,
                                                  batch_config_decoder,
                                                  max_workspace_size,
                                                  build_config_decoder,
                                                  encoder_input_size,
                                                  encoder_output_sizes);
}

void SAM2Image::RunEncoder(const std::vector<cv::Mat>& images)
{
    // Clear all variables
    masks_.clear();
    orig_im_size_.clear();

    // Run encoder to get results
    encoder_->EncodeImage(images);

    for (const auto& image : images)
    {
        orig_im_size_.push_back(image.size());
    }
}

void SAM2Image::RunDecoder(const std::vector<std::vector<cv::Rect>>& boxes)
{
    assert(boxes.size() == orig_im_size_.size());
    for (size_t i = 0; i < boxes.size(); i++)
    {
        auto boxes_per_image = boxes[i];
        std::vector<cv::Mat> masks_per_image;

        for (int z = 0; z < int(boxes_per_image.size()); z += decoder_batch_limit_)
        {
            int current_batch_size =
                std::min(decoder_batch_limit_, int(boxes_per_image.size()) - z);
            ClearBoxes();
            // Pre-allocate local storage
            std::vector<std::vector<cv::Point2f>> local_box_coords(current_batch_size);
            std::vector<std::vector<float>> local_box_labels(current_batch_size);

// Generate box information in parallel
#pragma omp parallel for
            for (int j = 0; j < current_batch_size; j++)
            {
                const auto& box = boxes_per_image[z + j];
                // Calculate two corner points of the box
                std::vector<cv::Point2f> coords = {
                    cv::Point2f(box.x, box.y), cv::Point2f(box.x + box.width, box.y + box.height)};
                // Label data for top-left and bottom-right corners of bbox
                std::vector<float> labels = {2, 3};

                local_box_coords[j] = coords;
                local_box_labels[j] = labels;
            }

            // Merge local results into member variables
            box_coords_ = std::move(local_box_coords);
            box_labels_ = std::move(local_box_labels);

            DecodeMask(orig_im_size_[i], i, masks_per_image, current_batch_size);
        }
        masks_.push_back(masks_per_image);
    }
}

void SAM2Image::DecodeMask(const cv::Size& orig_im_size,
                           const int img_batch_idx,
                           std::vector<cv::Mat>& masks_per_image,
                           const int current_batch_size)
{
    decoder_->Predict(encoder_->embed_data,
                      encoder_->feats_0_data,
                      encoder_->feats_1_data,
                      box_coords_,
                      box_labels_,
                      orig_im_size,
                      img_batch_idx,
                      current_batch_size);
    auto masks_per_image_per_decoder_batch = decoder_->result_masks;
    masks_per_image.insert(masks_per_image.end(),
                           masks_per_image_per_decoder_batch.begin(),
                           masks_per_image_per_decoder_batch.end());
}

const std::vector<std::vector<cv::Mat>>& SAM2Image::GetMasks()
{
    return masks_;
}

void SAM2Image::ClearBoxes()
{
    box_coords_.clear();
    box_labels_.clear();
}

void SAM2Image::GetMaxEntropy(cv::Mat& output_image, float& entropy_score)
{
    if (masks_.empty() || masks_[0].empty()) {
        output_image = cv::Mat::zeros(256, 256, CV_8UC3);
        entropy_score = 0.0f;
        return;
    }

    // Calculate average of all masks
    cv::Mat avg_mask = cv::Mat::zeros(masks_[0][0].size(), CV_32F);
    int count = 0;
    
    for (const auto& image_masks : masks_) {
        for (const auto& mask : image_masks) {
            cv::Mat float_mask;
            mask.convertTo(float_mask, CV_32F, 1.0/255.0);
            avg_mask += float_mask;
            count++;
        }
    }
    
    if (count > 0) {
        avg_mask /= count;
    }

    // Calculate entropy map
    cv::Mat entropy_map = cv::Mat::zeros(avg_mask.size(), CV_32F);
    for (int i = 0; i < avg_mask.rows; i++) {
        for (int j = 0; j < avg_mask.cols; j++) {
            float p = avg_mask.at<float>(i, j);
            if (p > 0 && p < 1) {
                float entropy = -p * std::log2(p) - (1-p) * std::log2(1-p);
                entropy_map.at<float>(i, j) = entropy;
            }
        }
    }

    // Calculate total entropy score
    entropy_score = cv::mean(entropy_map)[0];

    // Convert entropy map to visualization image
    cv::Mat normalized_entropy;
    cv::normalize(entropy_map, normalized_entropy, 0, 255, cv::NORM_MINMAX);
    normalized_entropy.convertTo(output_image, CV_8U);
    cv::applyColorMap(output_image, output_image, cv::COLORMAP_JET);
}