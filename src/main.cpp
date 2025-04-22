/**
 * @file main.cpp
 * @brief Main entry point for SAM2 TensorRT inference
 * 
 * Copyright (c) 2024 TIERIV
 * Author: Hunter Cheng (haoxuan.cheng@tier4.jp)
 * Created: 2025.4
 */

#include "argparse/argparse.hpp" // Ensure you have the argparse library in your include path
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "sam2_image_inference.hpp"
#include "utils.hpp"

void ProcessImage(std::string &encoder_path, std::string &decoder_path,
                  std::string &img_path, std::string &bbox_file_path,
                  std::string &output_jpg_path, std::string &precision,
                  const size_t batch_size, const int decoder_batch_limit) {
  // Get image and bbox filenames
  std::vector<std::string> image_names;

  for (const auto &entry : std::filesystem::directory_iterator(img_path)) {
    image_names.push_back(entry.path().string());
  }

  // Create SAM2Image object
  std::unique_ptr<SAM2Image> sam2;
  cv::Size encoder_input_size(1024, 1024); // Current encoder input size, affects decoder normalization
  sam2 = std::make_unique<SAM2Image>(encoder_path, decoder_path,
                                     encoder_input_size, precision,
                                     decoder_batch_limit);

  for (size_t i = 0; i < image_names.size(); i += batch_size) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> images_batch;
    std::vector<std::vector<cv::Rect>> box_coords_batch;
    // Calculate actual batch size for this iteration
    size_t current_batch_size = std::min(batch_size, image_names.size() - i);

    // Read images and bounding boxes
    for (size_t j = 0; j < current_batch_size; j++) {
      std::filesystem::path image_path = image_names[i + j];
      std::string image_file_name = image_path.filename().string();
      std::string bb_file_name;
      if (image_file_name.find(".jpg") != std::string::npos) {
        bb_file_name = replaceOtherString(image_file_name, ".jpg", ".txt");
      } else if (image_file_name.find(".png") != std::string::npos) {
        bb_file_name = replaceOtherString(image_file_name, ".png", ".txt");
      }

      // Read image and bounding box
      std::filesystem::path bb_file_path =
          std::filesystem::path(bbox_file_path) / bb_file_name;
      images_batch.push_back(cv::imread(image_path.string()));
      std::vector<cv::Rect> box_coords =
          read_and_transform_coordinates(bb_file_path.string());
      box_coords_batch.push_back(box_coords);
    }

    // Run encoder
    auto start_encoder = std::chrono::high_resolution_clock::now();
    sam2->RunEncoder(images_batch);
    auto end_encoder = std::chrono::high_resolution_clock::now();

    // Run decoder
    auto start_decoder = std::chrono::high_resolution_clock::now();
    sam2->RunDecoder(box_coords_batch);
    auto end_decoder = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<cv::Mat>> masks = sam2->GetMasks();

    auto start_draw = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < current_batch_size; j++) {
      cv::Mat masked_img = draw_masks(images_batch[j], masks[j]);
      cv::imwrite(output_jpg_path + "_" + std::to_string(i + j) + ".jpg",
                  masked_img);
    }
    auto end_draw = std::chrono::high_resolution_clock::now();

    auto duration_encoder =
        std::chrono::duration<double>(end_encoder - start_encoder);
    std::cout << "Encoder time: " << duration_encoder.count() << "s"
              << std::endl;
    auto duration_decoder =
        std::chrono::duration<double>(end_decoder - start_decoder);
    std::cout << "Decoder time: " << duration_decoder.count() << "s"
              << std::endl;
    auto duration_draw =
        std::chrono::duration<double>(end_draw - start_draw);
    std::cout << "Draw time: " << duration_draw.count() << "s" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    std::cout << "Total time(one iteration): " << duration.count() << "s"
              << std::endl;
  }
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("sam2_ort_cpp");

  // Define positional arguments
  program.add_argument("encoder_path")
      .help("Path to the encoder ONNX model file");

  program.add_argument("decoder_path")
      .help("Path to the decoder ONNX model file");

  program.add_argument("img_folder_path")
      .help("Path to the input images folder");

  program.add_argument("bbox_file_folder_path")
      .help("Path to the bounding box files' folder");

  program.add_argument("output_folder_path")
      .help("Path to folder for saving the output image file");

  // Define optional arguments
  program.add_argument("--precision")
      .help("Model precision (e.g., fp32 or fp16)")
      .default_value(std::string("fp32"));

  program.add_argument("--batch_size")
      .help("Batch size")
      .default_value(static_cast<size_t>(1))
      .scan<'i', size_t>();

  program.add_argument("--decoder_batch_limit")
      .help("Decoder batch limit")
      .default_value(static_cast<int>(50))
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);

    // Get arguments
    std::string encoder_path = program.get<std::string>("encoder_path");
    std::string decoder_path = program.get<std::string>("decoder_path");
    std::string img_path = program.get<std::string>("img_folder_path");
    std::string bbox_file_path =
        program.get<std::string>("bbox_file_folder_path");
    std::string output_jpg_path =
        program.get<std::string>("output_folder_path");
    std::string precision = program.get<std::string>("--precision");
    size_t batch_size = program.get<size_t>("--batch_size");
    int decoder_batch_limit = program.get<int>("--decoder_batch_limit");

    ProcessImage(encoder_path, decoder_path, img_path, bbox_file_path,
                 output_jpg_path, precision, batch_size, decoder_batch_limit);

  } catch (const std::exception &err) {
    std::cerr << "Error parsing arguments: " << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  return 0;
}
