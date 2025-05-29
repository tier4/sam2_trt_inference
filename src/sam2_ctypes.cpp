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

#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
using json = nlohmann::json;
#include "sam2_image_inference.hpp"
#include "utils.hpp"

extern "C" void InitOpenCVThreads()
{
    cv::setNumThreads(1);
}

extern "C"
{
    void* create_sam2image(const char* encoder_path,
                           const char* decoder_path,
                           const char* precision,
                           const int decoder_batch_limit)
    {
        cv::Size encoder_input_size(
            1024, 1024);  //当前的encoder的输入尺寸，关系到decoder中的normalization
        auto sam2 = new std::shared_ptr<SAM2Image>(std::make_unique<SAM2Image>(
            encoder_path, decoder_path, encoder_input_size, precision, decoder_batch_limit));
        return static_cast<void*>(sam2);
    }
    // Set image in SAM2Image
    void sam2image_set_image(void* instance, const unsigned char* image_data, int width, int height)
    {
        if (!instance || !image_data)
            return;
        cv::Mat image(height, width, CV_8UC3, (void*)image_data);

        auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
        sam2->RunEncoder({image});
    }

    // Set bounding box in SAM2Image
    void sam2image_set_box(void* instance, cv::Rect* boxes, int num_boxes)
    {
        std::vector<cv::Rect> box_coords_batch(boxes, boxes + num_boxes);
        if (!instance)
            return;
        auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
        sam2->RunDecoder({box_coords_batch});
    }

    // Get masks from SAM2Image
    void sam2image_get_masks(void* instance, unsigned char* output_image, int width, int height)
    {
        if (!instance || !output_image)
            return;
        auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
        std::vector<std::vector<cv::Mat>> masks = sam2->GetMasks();
        cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
        cv::Mat result = DrawMasks(img, masks[0], 0.5, true);
        memcpy(output_image, result.data, width * height * 3);
    }

    // Get max entropy map
    void sam2image_get_max_entropy(void* instance,
                                   unsigned char* output_image,
                                   int width,
                                   int height,
                                   float* entropy_score)
    {
        if (!instance || !output_image || !entropy_score)
            return;

        auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
        float peak_entropy = 0.0;
        cv::Mat max_entropy_map = sam2->GetMaxEntropy(peak_entropy);

        if (!max_entropy_map.empty())
        {
            memcpy(output_image, max_entropy_map.data, width * height * 3);
        }

        *entropy_score = peak_entropy;
    }

    const char* sam2image_get_polygon_str(void* instance,
                                          int width,
                                          int height,
                                          const char** names_array,
                                          int length,
                                          float* prob_array,
                                          int* id_array,
                                          const char* image_name,
                                          float peak_entropy_score)
    {
        static std::string json_string;
        if (!instance)
            return NULL;
        std::vector<std::string> names;
        for (int i = 0; i < length; i++)
        {
            if (names_array[i])
            {
                names.push_back(std::string(names_array[i]));
            }
        }
        std::string filename(image_name);
        auto sam2image = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
        std::vector<std::vector<cv::Mat>> masks = sam2image->GetMasks();
        std::vector<float> entropies = sam2image->GetEntropies();
        json::object_t imageAnnotationsOrdered;
        json::object_t uncertainty;
        imageAnnotationsOrdered["name"] = filename;
        imageAnnotationsOrdered["width"] = width;
        imageAnnotationsOrdered["height"] = height;
        uncertainty["instance"] = peak_entropy_score;
        imageAnnotationsOrdered["uncertainty"] = uncertainty;
        imageAnnotationsOrdered["annotations"] = json::array();
        for (size_t m = 0; m < masks[0].size(); m++)
        {
            if (masks[0][m].empty())
            {
                continue;
            }

            std::vector<std::vector<cv::Point>> contours = get_polygons(masks[0][m]);

            json::object_t annotationOrdered;
            annotationOrdered["type"] = "segmentation";
            annotationOrdered["title"] = names[m];
            annotationOrdered["value"] = names[m];
            annotationOrdered["prob"] = prob_array[m];
            annotationOrdered["id"] = id_array[m];
            annotationOrdered["uncertainty"] = entropies[m];

            if (contours.size() == 0)
                continue;
            for (size_t i = 0; i < contours.size(); ++i)
            {
                json contourJson;
                json pointsArray = json::array();
                for (size_t j = 0; j < contours[i].size(); ++j)
                {
                    pointsArray.push_back(contours[i][j].x);
                    pointsArray.push_back(contours[i][j].y);
                }
                annotationOrdered["points"].push_back({pointsArray});
            }
            imageAnnotationsOrdered["annotations"].push_back(annotationOrdered);
        }
        json_string = json(imageAnnotationsOrdered).dump();
        return json_string.c_str();
    }
}
