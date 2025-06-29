import os
ROOT_DIR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), ".."
    )
)

MMSEG_DIR_PATH = os.path.join(ROOT_DIR_PATH, "mmsegmentation")
MMSEG_MODEL_CONFIG_DIR_PATH = os.path.join(MMSEG_DIR_PATH, "configs")

SRC_DIR_PATH = os.path.join(ROOT_DIR_PATH, "src")
CELEBA_PIPELINE_PATH = os.path.join(SRC_DIR_PATH, "celebamaskhq_pipeline_512x512.py")

ASSETS_DIR_PATH = os.path.join(ROOT_DIR_PATH, "assets")
MODEL_CONFIG_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "model_configs")
CONFIG_FASTSCNN_DIR_PATH = os.path.join(MODEL_CONFIG_DIR_PATH, "fastscnn")
CONFIG_SEGFORMER_MITB5_DIR_PATH = os.path.join(MODEL_CONFIG_DIR_PATH, "segformer_mit-b5")
CONFIG_MOBILENETV3_LSRAPP_DIR_PATH = os.path.join(MODEL_CONFIG_DIR_PATH, "mobilenetv3_lraspp")
CONFIG_BISENETV2_FCN_DIR_PATH = os.path.join(MODEL_CONFIG_DIR_PATH, "bisenetv2_fcn")

MODEL_CONFIGS_DIRS = [
    CONFIG_FASTSCNN_DIR_PATH,
    CONFIG_SEGFORMER_MITB5_DIR_PATH,
    CONFIG_MOBILENETV3_LSRAPP_DIR_PATH,
    CONFIG_BISENETV2_FCN_DIR_PATH,
]
for mcd in MODEL_CONFIGS_DIRS:
    if not os.path.exists(mcd):
        os.makedirs(mcd)

OUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, "out")
OUT_FASTSCNN_DIR_PATH = os.path.join(OUT_DIR_PATH, "fastscnn")
OUT_SEGFORMER_MITB5_DIR_PATH = os.path.join(OUT_DIR_PATH, "segformer_mit-b5")
OUT_MOBILENETV3_LSRAPP_DIR_PATH = os.path.join(OUT_DIR_PATH, "mobilenetv3_lraspp")
OUT_BISENETV2_FCN_DIR_PATH = os.path.join(OUT_DIR_PATH, "bisenetv2_fcn")

OUT_MODELS_DIRS = [
    OUT_FASTSCNN_DIR_PATH,
    OUT_SEGFORMER_MITB5_DIR_PATH,
    OUT_MOBILENETV3_LSRAPP_DIR_PATH,
    OUT_BISENETV2_FCN_DIR_PATH,
]
for od in OUT_MODELS_DIRS:
    if not os.path.exists(od):
        os.makedirs(od)

DATASET_DIR_PATH = os.path.join(ROOT_DIR_PATH, "dev-public-fixed")
ORIGIN_IMG_DIR_PATH = os.path.join(DATASET_DIR_PATH, "train", "images")
ORIGIN_ANNO_DIR_PATH = os.path.join(DATASET_DIR_PATH, "train", "masks")
IMG_DIR_PATH = os.path.join(DATASET_DIR_PATH, "img_dir")
ANNO_DIR_PATH = os.path.join(DATASET_DIR_PATH, "ann_dir")
IMG_TRAIN_DIR_PATH = os.path.join(IMG_DIR_PATH, "train")
IMG_VAL_DIR_PATH = os.path.join(IMG_DIR_PATH, "val")
ANNO_TRAIN_DIR_PATH = os.path.join(ANNO_DIR_PATH, "train")
ANNO_VAL_DIR_PATH = os.path.join(ANNO_DIR_PATH, "val")
IMG_TEST_DIR_PATH = os.path.join(DATASET_DIR_PATH, "test", "images")
FINAL_IMG_TEST_DIR_PATH = os.path.join(DATASET_DIR_PATH, "test-public", "test", "images")

MASK_DIR_PATH = os.path.join(ROOT_DIR_PATH, "masks")
SOLUTION_DIR_PATH = os.path.join(ROOT_DIR_PATH, "solution")

fastscnn_timestamp = "20250325_202622"
fastscnn_best_iter = 10500

#mobilenetv3_timestamp = "20250325_220924" #latest

mobilenetv3_timestamp = "20250323_003538"
mobilenetv3_timestamp = "20250324_190555" #best?0.77
mobilenetv3_best_iter = 13000

# #1
# bisenetv2_timestamp = "20250326_094633"
# bisenetv2_best_iter = 16000

#2
# bisenetv2_timestamp = "20250326_170145"
# bisenetv2_best_iter = 18500

# # #3 only CE
# # bisenetv2_timestamp = "20250326_121549"
# # bisenetv2_best_iter = 11000

# # bisenetv2_timestamp = "20250326_131039" # all agumentations, full version of model, 20k
# # bisenetv2_best_iter = 18000

# bisenetv2_timestamp = "20250327_012148" #reforged
# bisenetv2_best_iter = 18000

bisenetv2_timestamp  = "20250327_030322"
bisenetv2_best_iter = 22500


CKPT_CONFIG_PATH = os.path.join(OUT_BISENETV2_FCN_DIR_PATH, f"{bisenetv2_timestamp}/vis_data", "config.py")

CKPT_CONFIG_LOOKUPS = {
    "ckpt.pth":{
        "config": CKPT_CONFIG_PATH,
        "checkpoint": os.path.join(SOLUTION_DIR_PATH, "ckpt.pth")
    },
    # "segformer_mitb5.pth":{
    #     "config": os.path.join(OUT_SEGFORMER_MITB5_DIR_PATH, "20250322_201422/celeba_segformer_mitb5_20250322.py"),
    #     "checkpoint": os.path.join(OUT_SEGFORMER_MITB5_DIR_PATH, "20250322_201422/best_mFscore_iter_18000.pth")
    # },
    "mobilnetv3_lraspp.pth":{
        # "config": os.path.join(OUT_MOBILENETV3_LSRAPP_DIR_PATH, "20250323_003538/celeba_mobilenetv3_lsrapp_20250323.py"),
        # "checkpoint": os.path.join(OUT_MOBILENETV3_LSRAPP_DIR_PATH, "20250323_003538/best_mFscore_iter_13000.pth")
        "config": os.path.join(OUT_MOBILENETV3_LSRAPP_DIR_PATH, f"{mobilenetv3_timestamp}/vis_data", "config.py"),
        "checkpoint": os.path.join(OUT_MOBILENETV3_LSRAPP_DIR_PATH, f"{mobilenetv3_timestamp}/best_mFscore_iter_{mobilenetv3_best_iter}.pth"),
    },
    "fastscnn.pth":{
        "config": os.path.join(OUT_FASTSCNN_DIR_PATH, f"{fastscnn_timestamp}/vis_data", "config.py"),
        "checkpoint": os.path.join(OUT_FASTSCNN_DIR_PATH, f"{fastscnn_timestamp}/best_mFscore_iter_{fastscnn_best_iter}.pth"),
    },
    "bisenetv2_fcn.pth":{
        "config": os.path.join(OUT_BISENETV2_FCN_DIR_PATH, f"{bisenetv2_timestamp}/vis_data", "config.py"),
        "checkpoint": os.path.join(OUT_BISENETV2_FCN_DIR_PATH, f"{bisenetv2_timestamp}/best_mIoU_iter_{bisenetv2_best_iter}.pth"),
    },
}


NUM_CLASS = 19
CROP_SIZE = (512, 512)
IMG_MEAN = [130.414, 104.74854, 91.30021] # RGB order
IMG_STD = [68.61917, 61.35547, 59.26968] # RGB order
NORM_CFG = dict(type='BN', requires_grad=True)
# class_weight=[
#     0.95, 0.95, 0.95, 1.0, 1.0, 1.0,
#     1.05, 1.05, 1.05, 1.0, 1.0, 1.0,
#     0.95, 0.95, 1.05, 1.5, 2.0, 1.0, 1.05
# ],
class_weights=[
    0.85,  # Background (大面积，进一步降低权重)
    0.90,  # Skin (合理)
    1.00,  # Nose (定位重要，稍微提高)
    1.80,  # Eye glasses (出现少且易混淆，更突出)
    1.00,  # Left eye (合理)
    1.00,  # Right eye (合理)
    1.20,  # Left brow (合理)
    1.20,  # Right brow (合理)
    1.50,  # Left ear (进一步突出小区域)
    1.50,  # Right ear (同上)
    1.00,  # Mouth (合理)
    1.20,  # Upper lip (合理)
    1.20,  # Lower lip (合理)
    0.90,  # Hair (合理)
    1.50,  # Hat (更突出，帽子易被混淆)
    2.00,  # Ear ring (合理)
    2.25,  # Necklace (略降，避免过高权重干扰)
    1.00,  # Neck (合理)
    1.00   # Cloth (合理)
]


DATA_PREPROCESSOR =  dict(
    type='SegDataPreProcessor',
    mean=IMG_MEAN,
    std=IMG_STD,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size= CROP_SIZE
)

weighted_cross_lovasz_loss_decode = [
    dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.5,
        class_weight=class_weights,
    ),
    dict(
        type='LovaszLoss',
        loss_type='multi_class',
        reduction='none',
        loss_weight=1.0,
        loss_name='loss_lovasz'
    )
]

weighted_cross_lovasz_loss_decode_auxilary = [
    dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.2,
        class_weight=class_weights,
    ),
    dict(
        type='LovaszLoss',
        loss_type='multi_class',
        reduction='none',
        loss_weight=0.4,
        loss_name='loss_lovasz'
    )
]

weighted_CE_dice_loss_decode = [
    dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.5,
        class_weight=class_weights
    ),
    dict(
        type='DiceLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        loss_name='loss_dice'
    )
]

weighted_CE_dice_loss_decode_auxiliary = [
    dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.2,
        class_weight= class_weights,
    ),
    dict(
        type='DiceLoss',
        use_sigmoid=False,
        loss_weight=0.4,
        loss_name='loss_dice'
    )
]

weighted_CE_lovasz_dice_loss_decode = [
        dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.4,
        class_weight=class_weights
    ),
    dict(
        type='DiceLoss',
        use_sigmoid=False,
        loss_weight=0.2,
        loss_name='loss_dice'
    ),
    dict(
        type='LovaszLoss',
        loss_type='multi_class',
        reduction='none',
        loss_weight=1.0,
        loss_name='loss_lovasz'
    )
]

weighted_CE_lovasz_dice_loss_decode_auxiliary = [
        dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.0533,
        class_weight= class_weights,
    ),
    dict(
        type='DiceLoss',
        use_sigmoid=False,
        loss_weight=0.2133,
        loss_name='loss_dice'
    ),
    dict(
        type='LovaszLoss',
        loss_type='multi_class',
        reduction='none',
        loss_weight=0.1333,
        loss_name='loss_lovasz'
    )
]

mitb5_fcn_auxiliary_head =  dict(
    type='FCNHead',
    in_channels=128,       # [64,128,320,512], select idx 1
    in_index=1,            
    channels=64,
    num_convs=1,
    num_classes=NUM_CLASS,
    norm_cfg=NORM_CFG,
    align_corners=False,
    loss_decode= weighted_cross_lovasz_loss_decode_auxilary
)

mobilenetv3_fcn_auxiliary_head =  dict(
    type='FCNHead',
    in_channels=24,       # [16,16,24,40....]
    in_index=2,            
    channels=64,
    num_convs=1,
    num_classes=NUM_CLASS,
    norm_cfg=NORM_CFG,
    align_corners=False,
    loss_decode= weighted_CE_lovasz_dice_loss_decode_auxiliary
)

adamw_optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-3,            
        weight_decay=1e-4,
    ),
    clip_grad=None
)

fastscnn_param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,         # 最小学习率
        begin=0,
        end=16000,           # 保持原来 step 数不变
        T_max=16000,         # CosineAnnealing 的周期
        by_epoch=False
    )
]



# from pathlib import Path

# directory = Path(TEST_IMG_DIR_PATH)
# extension = "jpg"

# count = len(list(directory.glob(f"*.{extension}")))
# print(f"Found {count} '{extension}' files in {directory}")
# from models_nparams_dict import filtered_model_types_nflops_nparams
# print(filtered_model_types_nflops_nparams)