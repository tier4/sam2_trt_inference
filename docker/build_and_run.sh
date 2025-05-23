#!/bin/bash

# Set image and container name
IMAGE_NAME=sam2_trt_env
CONTAINER_NAME=sam2_trt_container

# Build the Docker image
echo "🔧 Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Run the Docker container
echo "🚀 Running Docker container: $CONTAINER_NAME..."
docker run --gpus all -it --rm \
  --shm-size=16g \
  --name $CONTAINER_NAME \
  -v $(pwd)/..:/workspace/sam2_trt_cpp \
  $IMAGE_NAME bash
