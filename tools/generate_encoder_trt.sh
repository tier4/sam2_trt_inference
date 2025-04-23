#!/bin/bash

# Help function
print_usage() {
    echo "Usage: $0 <onnx_model_path> <output_engine_path> [options]"
    echo "Options:"
    echo "  --precision <type>  Model precision, fp16 or fp32 (default: fp16)"
    echo "  --workspace <size>  Workspace size in MB (default: 4096)"
    exit 1
}

# Check if minimum arguments are provided
if [ $# -lt 2 ]; then
    print_usage
fi

MODEL="$1"
ENGINE="$2"
shift 2

# Default values
PRECISION="fp16"
WORKSPACE=4096

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Define fixed inputs
FIXED_INPUTS=(
  "image:1x3x1024x1024"
)

# Build the command
SHAPES=""

# Process fixed inputs
for input in "${FIXED_INPUTS[@]}"; do
  IFS=':' read -r name shape <<< "$input"
  
  if [ -n "$SHAPES" ]; then
    SHAPES+=","
  fi
  
  SHAPES+="$name:$shape"
done

echo "Converting ONNX model: $MODEL"
echo "Output engine path: $ENGINE"
echo "Precision: ${PRECISION}"
echo "Workspace size: ${WORKSPACE}MB"
echo "Fixed inputs: ${FIXED_INPUTS[@]}"

# Prepare precision flag
PRECISION_FLAG=""
if [ "$PRECISION" = "fp16" ]; then
    PRECISION_FLAG="--fp16"
elif [ "$PRECISION" = "fp32" ]; then
    PRECISION_FLAG="--fp32"
else
    echo "Error: Invalid precision type. Must be fp16 or fp32"
    exit 1
fi

# Execute the command
trtexec --onnx="$MODEL" \
        --saveEngine="$ENGINE" \
        --shapes="$SHAPES" \
        ${PRECISION_FLAG} \
        --workspace=$WORKSPACE \
        --verbose 