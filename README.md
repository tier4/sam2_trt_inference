# SAM2 TensorRT C++ Inference

A high-performance TensorRT inference framework for Segment Anything Model 2 (SAM2) implemented in C++, with tools for model conversion from ONNX to TensorRT engine.
![SAM2 TensorRT C++ Inference](assets/thumbnail.png)

## Features

- **High-Performance Inference**: Optimized C++ implementation using TensorRT for fast SAM2 model inference
- **Batch Processing**: Support for batch processing to maximize throughput
- **OpenMP Acceleration**: Multi-threaded processing for CPU tasks using OpenMP
- **Flexible Input**: Process multiple images and bounding boxes in batch
- **Model Precision Options**: Support for FP16 and FP32 precision models
- **CUDA Optimizations**: Efficient GPU memory management with CUDA streams
- **Visualization Tools**: Utilities for visualizing segmentation results

## Requirements

- NVIDIA GPU with CUDA support
- Docker and NVIDIA Container Toolkit
- TensorRT (8.5 or 8.6)
- OpenCV
- Boost libraries
- CMake (3.10+)

## Environment Setup

### Using Docker (Recommended)

1. Pull the TensorRT Docker image:
```bash
docker pull nvcr.io/nvidia/tensorrt:24.01-py3
```

2. Launch the container with GPU support:
```bash
docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.01-py3
```

3. Install dependencies inside the container:
```bash
sudo apt update
sudo apt install libboost-all-dev
sudo apt install libopencv-dev
```

### Manual Setup

If not using Docker, ensure you have the following installed:
- CUDA Toolkit
- TensorRT (8.5 or 8.6)
- OpenCV
- Boost libraries
- OpenMP

## Building the Project

```bash
cmake -B build
cd build
make
```

## Usage

### Convert models from pytorch to onnx

Refer to this [repo](https://github.com/tier4/sam2_pytorch2onnx) for converting onnx models

### Running Inference with pre-generated TensorRT engine

Use the provided script to convert your SAM2 ONNX models to TensorRT format:

```bash
bash tools/generate_encoder_trt.sh path/to/encoder.onnx path/to/encoder.engine [options]
bash tools/generate_decoder_trt.sh path/to/decoder.onnx path/to/decoder.engine [options]
```

Options:
- `--min-batch <N>`: Minimum batch size (default: 1)
- `--opt-batch <N>`: Optimal batch size (default: 128)
- `--max-batch <N>`: Maximum batch size (default: 200)
- `--precision <fp16|fp32>`: Model precision (default: fp16)
- `--workspace <size>`: Workspace size in MB (default: 4096)

The encoder model uses a fixed batch size of 1, while the decoder model's batch size is dynamically configured based on your GPU capabilities and memory constraints.

```bash
./trtsam2 encoder.engine decoder.engine images_folder/ bboxes_folder/ output_folder/ [options]
```

### Running Inference with ONNX model

```bash
./trtsam2 encoder.onnx decoder.onnx images_folder/ bboxes_folder/ output_folder/ [options]
```

#### Command Line Arguments

- `encoder_path`: Path to the encoder TensorRT engine or ONNX model
- `decoder_path`: Path to the decoder TensorRT engine or ONNX model
- `img_folder_path`: Path to the folder containing input images
- `bbox_file_folder_path`: Path to the folder containing bounding box files
- `output_folder_path`: Path to save the segmentation results

#### Options

- `--precision <fp16|fp32>`: Model precision (default: fp32)
- `--decoder_batch_limit <N>`: Maximum batch size for decoder (default: 50)

### Input Format

The bounding box files should be in a text format with each line containing:
```
class_name confidence left top right bottom
```

Where:
- `class_name`: The class name of the object
- `confidence`: Detection confidence score (between 0 and 1)
- `left`: X coordinate of the top-left corner of the bounding box
- `top`: Y coordinate of the top-left corner of the bounding box
- `right`: X coordinate of the bottom-right corner of the bounding box
- `bottom`: Y coordinate of the bottom-right corner of the bounding box

This format is based on the [mAP (mean Average Precision)](https://github.com/Cartucho/mAP) evaluation tool.

### Input File Naming Convention

The image files and their corresponding bounding box files must have matching names (excluding extensions). For example:

```
images_folder/
    ├── image1.jpg
    ├── image2.png
    └── image3.jpeg

bboxes_folder/
    ├── image1.txt
    ├── image2.txt
    └── image3.txt
```

In this example:
- `image1.jpg` corresponds to `image1.txt`
- `image2.png` corresponds to `image2.txt`
- `image3.jpeg` corresponds to `image3.txt`

The program will process each image with its corresponding bounding box file based on the matching names. You can find sample data in the `sample_data` folder to test the inference.

## Benchmarks
- SAM2 base plus model
- 94 target boxes
- "whole" includes engine time, image I/O time, as well as pre-process and post-process time

| Device | Precision | Encoder (ms) | Decoder (ms) | Draw (ms) | Whole (ms) |
|--------|-----------|------------|--------------|--------------|------------|
| L40s | FP32 | 45 | 83 | 15 | 168 |
| L40s | FP16 | 23 | 63 | 13 | 123 |
| RTX 3070ti | FP16 | 60 | 276 | 46 | 414 |
| Jetson Orin | FP16 | 159 | 310 | 94 | 637 |


## License

This project is licensed under the Apache License 2.0

### Dependencies Licenses

- **SAM2**: Licensed under the Apache License 2.0
  - Original repository: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
  - Copyright (c) Meta Platforms, Inc. and affiliates.

- **argparse**: Licensed under the MIT License
  - Original repository: [p-ranav/argparse](https://github.com/p-ranav/argparse)
  - Copyright (c) 2018 Pranav Srinivas Kumar

## Acknowledgements

- [SAM2 Paper and Original Implementation](https://github.com/facebookresearch/sam2)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- OpenCV community
- [argparse](https://github.com/p-ranav/argparse)
