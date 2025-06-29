DATASET_DIR_PATH = '/home/featurize/work/AI6126project1/dev-public-fixed'
crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        130.414,
        104.74854,
        91.30021,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        68.61917,
        61.35547,
        59.26968,
    ],
    type='SegDataPreProcessor')
data_root = '/home/featurize/work/AI6126project1/dev-public-fixed'
dataset_type = 'CelebAMaskHQDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        rule='greater',
        save_best='mFscore',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=32,
            concat_input=False,
            in_channels=128,
            in_index=-2,
            loss_decode=[
                dict(
                    class_weight=[
                        0.95,
                        0.95,
                        0.95,
                        1.0,
                        1.0,
                        1.0,
                        1.05,
                        1.05,
                        1.05,
                        1.0,
                        1.0,
                        1.0,
                        0.95,
                        0.95,
                        1.05,
                        1.5,
                        2.0,
                        1.0,
                        1.05,
                    ],
                    loss_weight=0.04,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                dict(
                    loss_name='loss_dice',
                    loss_weight=0.1335,
                    type='DiceLoss',
                    use_sigmoid=False),
                dict(
                    loss_name='loss_lovasz',
                    loss_type='multi_class',
                    loss_weight=0.02665,
                    reduction='none',
                    type='LovaszLoss'),
            ],
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=19,
            num_convs=1,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=32,
            concat_input=False,
            in_channels=64,
            in_index=-3,
            loss_decode=[
                dict(
                    class_weight=[
                        0.95,
                        0.95,
                        0.95,
                        1.0,
                        1.0,
                        1.0,
                        1.05,
                        1.05,
                        1.05,
                        1.0,
                        1.0,
                        1.0,
                        0.95,
                        0.95,
                        1.05,
                        1.5,
                        2.0,
                        1.0,
                        1.05,
                    ],
                    loss_weight=0.04,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                dict(
                    loss_name='loss_dice',
                    loss_weight=0.1335,
                    type='DiceLoss',
                    use_sigmoid=False),
                dict(
                    loss_name='loss_lovasz',
                    loss_type='multi_class',
                    loss_weight=0.02665,
                    reduction='none',
                    type='LovaszLoss'),
            ],
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=19,
            num_convs=1,
            type='FCNHead'),
    ],
    backbone=dict(
        align_corners=False,
        downsample_dw_channels=(
            32,
            48,
        ),
        fusion_out_channels=128,
        global_block_channels=(
            64,
            96,
            128,
        ),
        global_block_strides=(
            2,
            2,
            1,
        ),
        global_in_channels=64,
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        norm_cfg=dict(requires_grad=True, type='BN'),
        out_indices=(
            0,
            1,
            2,
        ),
        type='FastSCNN'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            130.414,
            104.74854,
            91.30021,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            68.61917,
            61.35547,
            59.26968,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=128,
        concat_input=False,
        in_channels=128,
        in_index=-1,
        loss_decode=[
            dict(
                class_weight=[
                    0.95,
                    0.95,
                    0.95,
                    1.0,
                    1.0,
                    1.0,
                    1.05,
                    1.05,
                    1.05,
                    1.0,
                    1.0,
                    1.0,
                    0.95,
                    0.95,
                    1.05,
                    1.5,
                    2.0,
                    1.0,
                    1.05,
                ],
                loss_weight=0.2,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                loss_name='loss_dice',
                loss_weight=1.0,
                type='DiceLoss',
                use_sigmoid=False),
            dict(
                loss_name='loss_lovasz',
                loss_type='multi_class',
                loss_weight=0.3,
                reduction='none',
                type='LovaszLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=19,
        type='DepthwiseSeparableFCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.12, momentum=0.9, type='SGD', weight_decay=4e-05),
    type='OptimWrapper')
optimizer = dict(lr=0.12, momentum=0.9, type='SGD', weight_decay=4e-05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/featurize/work/AI6126project1/dev-public-fixed',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CelebAMaskHQDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=16000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='/home/featurize/work/AI6126project1/dev-public-fixed',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                degree=25,
                pad_val=0,
                prob=0.5,
                seg_pad_val=0,
                type='RandomRotate'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    1,
                    1.2,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(
                direction='horizontal',
                label_swap_map=dict({
                    4: 5,
                    6: 7,
                    8: 9
                }),
                prob=0.25,
                type='RandomFlipWithLabelSwap'),
            dict(
                brightness_delta=16,
                contrast_range=(
                    0.75,
                    1.25,
                ),
                hue_delta=9,
                saturation_range=(
                    0.75,
                    1.25,
                ),
                type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='CelebAMaskHQDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(degree=25, pad_val=0, prob=0.5, seg_pad_val=0, type='RandomRotate'),
    dict(
        keep_ratio=True,
        ratio_range=(
            1,
            1.2,
        ),
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(
        direction='horizontal',
        label_swap_map=dict({
            4: 5,
            6: 7,
            8: 9
        }),
        prob=0.25,
        type='RandomFlipWithLabelSwap'),
    dict(
        brightness_delta=16,
        contrast_range=(
            0.75,
            1.25,
        ),
        hue_delta=9,
        saturation_range=(
            0.75,
            1.25,
        ),
        type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='/home/featurize/work/AI6126project1/dev-public-fixed',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CelebAMaskHQDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
        'mDice',
    ], type='IoUMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/featurize/work/AI6126project1/out/fastscnn'
