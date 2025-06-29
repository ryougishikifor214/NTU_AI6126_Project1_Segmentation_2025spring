#!/bin/sh
conda env info
IMG_NAME="0a1a4d56a1744099bfa0b9cef8a00232.jpg"
WEIGHTS="ckpt.pth"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
RUN_SCRIPT_PATH=$(realpath "$SCRIPT_DIR/run.py")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
TEST_IMG_DIR=$(realpath "$PROJECT_ROOT/dev-public-fixed/test-public/test/images")
MASK_DIR=$(realpath "$PROJECT_ROOT/masks")

MASK_NAME="${IMG_NAME%.*}.png"
INPUT_IMG_PATH=$(realpath "$TEST_IMG_DIR/$IMG_NAME")
OUTPUT_MASK_PATH=$(realpath "$MASK_DIR/$MASK_NAME")

echo $RUN_SCRIPT_PATH
echo $INPUT_IMG_PATH
echo $OUTPUT_MASK_PATH

python "$RUN_SCRIPT_PATH"\
    --input "$INPUT_IMG_PATH"\
    --output "$OUTPUT_MASK_PATH"\
    --weights "$WEIGHTS"\