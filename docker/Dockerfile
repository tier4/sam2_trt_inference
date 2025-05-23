FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Set environment to non-interactive to avoid tzdata or other prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace