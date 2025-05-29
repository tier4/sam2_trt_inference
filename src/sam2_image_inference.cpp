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
#include "colormap.hpp"

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
    mat_entropies_.insert(
        mat_entropies_.end(), decoder_->mat_entropies_.begin(), decoder_->mat_entropies_.end());
    entropies_.insert(entropies_.end(), decoder_->entropies_.begin(), decoder_->entropies_.end());
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

cv::Mat SAM2Image::GetMaxEntropy(float& peak_entropy_score)
{
    cv::Mat max_ent, max_ent_jet;
    if (mat_entropies_.size() == 0)
    {
        return max_ent;
    }
    int height = mat_entropies_[0].rows;
    int width = mat_entropies_[0].cols;
    max_ent = cv::Mat::zeros(height, width, CV_8UC1);
    max_ent_jet = cv::Mat::zeros(height, width, CV_8UC3);

    for (int i = 0; i < (int)mat_entropies_.size(); i++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                max_ent.at<unsigned char>(y, x) =
                    (mat_entropies_[i].at<unsigned char>(y, x) > max_ent.at<unsigned char>(y, x))
                        ? mat_entropies_[i].at<unsigned char>(y, x)
                        : max_ent.at<unsigned char>(y, x);
            }
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const auto& color = jet_colormap[max_ent.at<unsigned char>(y, x)];
            max_ent_jet.at<cv::Vec3b>(y, x)[0] = color[0];
            max_ent_jet.at<cv::Vec3b>(y, x)[1] = color[1];
            max_ent_jet.at<cv::Vec3b>(y, x)[2] = color[2];
        }
    }
    float sum_ent = 0.0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            sum_ent += max_ent.at<unsigned char>(y, x);
        }
    }
    sum_ent /= (width * height);
    cv::putText(max_ent_jet,
                std::to_string(sum_ent),
                cv::Point(32, 32),
                1,
                1.0,
                cv::Scalar(255, 255, 255),
                1);
    peak_entropy_score = sum_ent;
    // return max_ent;
    return max_ent_jet;
}

const std::vector<float>& SAM2Image::GetEntropies() const
{
    return entropies_;
}