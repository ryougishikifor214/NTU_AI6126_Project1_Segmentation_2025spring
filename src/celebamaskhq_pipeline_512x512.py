
from global_config import DATASET_DIR_PATH

dataset_type = "CelebAMaskHQDataset"
data_root = DATASET_DIR_PATH
crop_size = (512, 512)

albu_train_transforms = [
    dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8), p=0.6),
    dict(type='Sharpen', alpha=(0.2, 0.5), lightness=(0.8, 1.2), p=0.5),
    dict(type='GaussianBlur', blur_limit=(3, 5), p=0.2),
    dict(type='GaussNoise', var_limit=(5,15), p=0.3),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    #dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlipWithLabelSwap', prob=0.5, direction='horizontal', label_swap_map={4:5, 6:7, 8:9}),
    dict(type='RandomRotate', prob=0.5, degree=15, pad_val=0, seg_pad_val=0),
    dict(type='RandomResize', scale=(640, 640), ratio_range=(0.8, 1.1), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.7),
    # dict(type="RandomFlip", prob=0.25, direction='horizontal'),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={'img': 'image', 'gt_semantic_seg': 'mask'},
        update_pad_shape=False
    ),
    dict(type='PhotoMetricDistortion', brightness_delta=16, contrast_range=(0.75, 1.25), saturation_range=(0.75, 1.25), hue_delta=9),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=val_pipeline)
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore', "mIoU", "mDice"])
test_evaluator = val_evaluator