#!/bin/sh
conda activate base
conda env list

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
MODELS_CONFIGS_DIR=$(realpath "$PROJECT_ROOT/assets/model_configs")
CUR_MODEL_CFG_SUFFIX="segformer_mit-b5/celeba_segformer_mitb5_20250327.py"
#CUR_MODEL_CFG_SUFFIX="fastscnn/celeba_fastscnn_20250325.py"
#CUR_MODEL_CFG_SUFFIX="mobilenetv3_lraspp/celeba_mobilenetv3_lsrapp_20250325.py"
#CUR_MODEL_CFG_SUFFIX="bisenetv2_fcn/celeba_bisenetv2_fcn_20250327.py"

CUR_MODEL_CFG=$(realpath "$MODELS_CONFIGS_DIR/$CUR_MODEL_CFG_SUFFIX")

MMSEG_TRAIN_TOOL=$(realpath "$PROJECT_ROOT/mmsegmentation/tools/train.py")

python $MMSEG_TRAIN_TOOL $CUR_MODEL_CFG