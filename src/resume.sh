#!/bin/sh
conda activate base
conda env list

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
#WORK_DIR=$(realpath "$PROJECT_ROOT/out/fastscnn/")
WORK_DIR=$(realpath "$PROJECT_ROOT/out/bisenetv2_fcn/")

CUR_MODEL_CFG_SUFFIX="20250326_160026/vis_data/config.py"

CUR_MODEL_CFG=$(realpath "$WORK_DIR/$CUR_MODEL_CFG_SUFFIX")
MMSEG_TRAIN_TOOL=$(realpath "$PROJECT_ROOT/mmsegmentation/tools/train.py")

python $MMSEG_TRAIN_TOOL $CUR_MODEL_CFG  --resume --work-dir=$WORK_DIR