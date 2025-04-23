# SAM2 TensorRT C++ Inference

A high-performance TensorRT inference framework for Segment Anything Model 2 (SAM2) implemented in C++, with tools for model conversion from ONNX to TensorRT engine.

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

```bash
waiting for development
```

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
./MyTRTSAM2App encoder.engine decoder.engine images_folder/ bboxes_folder/ output_folder/ [options]
```

### Running Inference with ONNX model

```bash
./MyTRTSAM2App encoder.onnx decoder.onnx images_folder/ bboxes_folder/ output_folder/ [options]
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
class_id confidence x_min y_min x_max y_max
```

## Architecture

### Components

- **SAM2ImageEncoder**: Processes input images to extract image embeddings
- **SAM2ImageDecoder**: Takes embeddings and prompts to generate masks
- **SAM2Image**: Main interface that coordinates the encoder and decoder
- **Utils**: Helper functions for visualization and data processing
- **TensorRTCommon**: Handles TensorRT engine loading, inference, and memory management

### Optimization Techniques

- **CUDA Streams**: Asynchronous operations for overlapping computation
- **OpenMP Parallelization**: Multi-threaded mask processing
- **Memory Management**: Custom CUDA memory allocation with proper synchronization
- **TensorRT Optimization**: Leveraging TensorRT for optimized inference

## License

[Your license information here]

## Acknowledgements

- [SAM2 Paper and Original Implementation](https://github.com/facebookresearch/sam2)
- NVIDIA for TensorRT
- OpenCV community
- [argparse](https://github.com/p-ranav/argparse)

## Contributing

[Contributing guidelines]