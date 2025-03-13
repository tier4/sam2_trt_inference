#!/bin/bash

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <onnx_model_path> <output_engine_path>"
    exit 1
fi

MODEL="$1"
ENGINE="$2"

# Define fixed inputs (these will not have dynamic batch size)
FIXED_INPUTS=(
  "image_embed:1x256x64x64"
  "high_res_feats_0:1x32x256x256"
  "high_res_feats_1:1x64x128x128"
  "has_mask_input:1"
)

# Define dynamic inputs (these will have dynamic batch size)
DYNAMIC_INPUTS=(
  "point_coords:1x2x2:50x2x2:100x2x2"
  "point_labels:1x2:50x2:100x2"
  "mask_input:1x1x256x256:50x1x256x256:100x1x256x256"
)

# Build the command
MIN_SHAPES=""
OPT_SHAPES=""
MAX_SHAPES=""

# Process dynamic inputs
for input in "${DYNAMIC_INPUTS[@]}"; do
  IFS=':' read -r name min opt max <<< "$input"
  
  if [ -n "$MIN_SHAPES" ]; then
    MIN_SHAPES+=","
    OPT_SHAPES+=","
    MAX_SHAPES+=","
  fi
  
  MIN_SHAPES+="$name:$min"
  OPT_SHAPES+="$name:$opt"
  MAX_SHAPES+="$name:$max"
done

# Process fixed inputs - they have the same shape for min, opt, and max
for input in "${FIXED_INPUTS[@]}"; do
  IFS=':' read -r name shape <<< "$input"
  
  if [ -n "$MIN_SHAPES" ]; then
    MIN_SHAPES+=","
    OPT_SHAPES+=","
    MAX_SHAPES+=","
  fi
  
  MIN_SHAPES+="$name:$shape"
  OPT_SHAPES+="$name:$shape"
  MAX_SHAPES+="$name:$shape"
done

echo "Converting ONNX model: $MODEL"
echo "Output engine path: $ENGINE"
echo "Fixed inputs: ${FIXED_INPUTS[@]}"
echo "Dynamic inputs: ${DYNAMIC_INPUTS[@]}"

# Execute the command
trtexec --onnx="$MODEL" \
        --saveEngine="$ENGINE" \
        --minShapes="$MIN_SHAPES" \
        --optShapes="$OPT_SHAPES" \
        --maxShapes="$MAX_SHAPES" \
        --fp16 \
        --workspace=4096 \
        --verbose