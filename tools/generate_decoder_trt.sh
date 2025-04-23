#!/bin/bash

# Help function
print_usage() {
    echo "Usage: $0 <onnx_model_path> <output_engine_path> [options]"
    echo "Options:"
    echo "  --min-batch <N>     Minimum batch size (default: 1)"
    echo "  --opt-batch <N>     Optimal batch size (default: 128)"
    echo "  --max-batch <N>     Maximum batch size (default: 200)"
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
MIN_BATCH=1
OPT_BATCH=128
MAX_BATCH=200
PRECISION="fp16"
WORKSPACE=4096

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-batch)
            MIN_BATCH="$2"
            shift 2
            ;;
        --opt-batch)
            OPT_BATCH="$2"
            shift 2
            ;;
        --max-batch)
            MAX_BATCH="$2"
            shift 2
            ;;
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

# Define fixed inputs (these will not have dynamic batch size)
FIXED_INPUTS=(
  "image_embed:1x256x64x64"
  "high_res_feats_0:1x32x256x256"
  "high_res_feats_1:1x64x128x128"
  "has_mask_input:1"
)

# Define dynamic inputs with user-specified batch sizes
DYNAMIC_INPUTS=(
  "point_coords:${MIN_BATCH}x2x2:${OPT_BATCH}x2x2:${MAX_BATCH}x2x2"
  "point_labels:${MIN_BATCH}x2:${OPT_BATCH}x2:${MAX_BATCH}x2"
  "mask_input:${MIN_BATCH}x1x256x256:${OPT_BATCH}x1x256x256:${MAX_BATCH}x1x256x256"
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
echo "Batch size (min/opt/max): ${MIN_BATCH}/${OPT_BATCH}/${MAX_BATCH}"
echo "Precision: ${PRECISION}"
echo "Workspace size: ${WORKSPACE}MB"
echo "Fixed inputs: ${FIXED_INPUTS[@]}"
echo "Dynamic inputs: ${DYNAMIC_INPUTS[@]}"

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
        --minShapes="$MIN_SHAPES" \
        --optShapes="$OPT_SHAPES" \
        --maxShapes="$MAX_SHAPES" \
        ${PRECISION_FLAG} \
        --workspace=$WORKSPACE \
        --verbose