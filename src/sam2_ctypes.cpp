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
#include <vector>
#include <map>
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "sam2_image_inference.hpp"
#include "utils.hpp"

extern "C" void InitOpenCVThreads() {
  cv::setNumThreads(1);
}

extern "C" {
  void* create_sam2image(const char* encoder_path, const char* decoder_path, const char* precision, const int decoder_batch_limit)
  {
    cv::Size encoder_input_size(1024, 1024); //当前的encoder的输入尺寸，关系到decoder中的normalization
    auto sam2 = new std::shared_ptr<SAM2Image>(
                       std::make_unique<SAM2Image>(encoder_path, decoder_path, encoder_input_size, precision, decoder_batch_limit));
    return static_cast<void*>(sam2);
  }
  // Set image in SAM2Image
  void sam2image_set_image(void* instance, const unsigned char* image_data, int width, int height)
  {
    if (!instance || !image_data) return;
    cv::Mat image(height, width, CV_8UC3, (void*)image_data);
    
    auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);                
    sam2->RunEncoder({image});
  }

  // Set bounding box in SAM2Image
  void sam2image_set_box(void* instance, cv::Rect* boxes, int num_boxes)
  {  
    std::vector<cv::Rect> box_coords_batch(boxes, boxes + num_boxes);
    if (!instance) return;
    auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
    sam2->RunDecoder({box_coords_batch});
  }

  // Get masks from SAM2Image
  void sam2image_get_masks(void* instance, unsigned char* output_image, int width, int height) {
    if (!instance || !output_image) return;
    auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);    
    std::vector<std::vector<cv::Mat>> masks = sam2->GetMasks();
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);    
    cv::Mat result = DrawMasks(img, masks[0], 0.5, true);
    memcpy(output_image, result.data, width * height * 3);
  }

  // Get max entropy map
  void sam2image_get_max_entropy(void* instance, unsigned char* output_image, int width, int height, float* entropy_score)
  {
    if (!instance || !output_image || !entropy_score) return;

    auto sam2 = *static_cast<std::shared_ptr<SAM2Image>*>(instance);    
    cv::Mat max_entropy_map;
    sam2->GetMaxEntropy(max_entropy_map, *entropy_score);

    if (!max_entropy_map.empty()) {
      memcpy(output_image, max_entropy_map.data, width * height * 3);
    }
  }
  
  const char* sam2image_get_polygon_str(void* instance, int width, int height, const char** names_array, int length, float *prob_array, int *id_array, const char *image_name, float peak_entropy_score)
  {
    static std::string json_string; 
    if (!instance) return NULL;
    
    std::vector<std::string> names;
    std::vector<float> probs;
    std::vector<int> ids;
    
    for (int i = 0; i < length; i++) {
      if (names_array[i]) {
        names.push_back(std::string(names_array[i])); 
        probs.push_back(prob_array[i]);
        ids.push_back(id_array[i]);
      }
    }
    
    auto sam2image = *static_cast<std::shared_ptr<SAM2Image>*>(instance);
    std::vector<std::vector<cv::Mat>> masks = sam2image->GetMasks();
    
    json::object_t imageAnnotationsOrdered;
    json::object_t uncertainty; 
    imageAnnotationsOrdered["name"] = image_name;
    imageAnnotationsOrdered["width"] = width;
    imageAnnotationsOrdered["height"] = height;
    uncertainty["instance"] = peak_entropy_score;
    imageAnnotationsOrdered["uncertainty"] = uncertainty;    
    imageAnnotationsOrdered["annotations"] = json::array();

    for (size_t m = 0; m < masks[0].size(); m++) {
      if (masks[0][m].empty()) {
        continue;
      }

      // Get contours
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(masks[0][m], contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      // Simplify contours
      std::vector<std::vector<cv::Point>> simplified_contours;
      for (const auto& contour : contours) {
        std::vector<cv::Point> simplified;
        cv::approxPolyDP(contour, simplified, 2.0, true);
        if (simplified.size() >= 3) {  // Only keep polygons with at least 3 points
          simplified_contours.push_back(simplified);
        }
      }

      json::object_t annotationOrdered;
      annotationOrdered["type"] = "segmentation";
      annotationOrdered["title"] = names[m];
      annotationOrdered["value"] = names[m];
      annotationOrdered["prob"] = probs[m];
      annotationOrdered["id"] = ids[m];      
      annotationOrdered["uncertainty"] = peak_entropy_score;

      if (simplified_contours.empty()) continue;
      
      for (const auto& contour : simplified_contours) {      
        json pointsArray = json::array();
        for (const auto& point : contour) {
          pointsArray.push_back(point.x);
          pointsArray.push_back(point.y);
        }
        annotationOrdered["points"].push_back({pointsArray});
      }
      imageAnnotationsOrdered["annotations"].push_back(annotationOrdered);
    }
    
    json_string = json(imageAnnotationsOrdered).dump();
    return json_string.c_str();    
  }
} 