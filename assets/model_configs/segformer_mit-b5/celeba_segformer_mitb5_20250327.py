DATASET_DIR_PATH = '/home/featurize/work/AI6126project1/dev-public-fixed'
albu_train_transforms = [
    dict(clip_limit=2.0, p=0.6, tile_grid_size=(
        8,
        8,
    ), type='CLAHE'),
    dict(alpha=(
        0.2,
        0.5,
    ), lightness=(
        0.8,
        1.2,
    ), p=0.5, type='Sharpen'),
    dict(blur_limit=(
        3,
        5,
    ), p=0.2, type='GaussianBlur'),
    dict(p=0.3, type='GaussNoise', var_limit=(
        5,
        15,
    )),
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
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
        max_keep_ckpts=2,
        rule='greater',
        save_best='mIoU',
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
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            3,
            6,
            40,
            3,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
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
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                class_weight=[
                    0.85,
                    0.9,
                    1.0,
                    1.8,
                    1.0,
                    1.0,
                    1.2,
                    1.2,
                    1.5,
                    1.5,
                    1.0,
                    1.2,
                    1.2,
                    0.9,
                    1.5,
                    2.0,
                    2.25,
                    1.0,
                    1.0,
                ],
                loss_weight=0.5,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                loss_name='loss_lovasz',
                loss_type='multi_class',
                loss_weight=1.0,
                reduction='none',
                type='LovaszLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=19,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.0075, momentum=0.9, type='SGD', weight_decay=0.001)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=24000,
        eta_min=5e-05,
        power=0.95,
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
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
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
        512,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=24000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='/home/featurize/work/AI6126project1/dev-public-fixed',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                direction='horizontal',
                label_swap_map=dict({
                    4: 5,
                    6: 7,
                    8: 9
                }),
                prob=0.5,
                type='RandomFlipWithLabelSwap'),
            dict(
                degree=15,
                pad_val=0,
                prob=0.5,
                seg_pad_val=0,
                type='RandomRotate'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.8,
                    1.1,
                ),
                scale=(
                    640,
                    640,
                ),
                type='RandomResize'),
            dict(cat_max_ratio=0.7, crop_size=(
                512,
                512,
            ), type='RandomCrop'),
            dict(
                keymap=dict(gt_semantic_seg='mask', img='image'),
                transforms=[
                    dict(
                        clip_limit=2.0,
                        p=0.6,
                        tile_grid_size=(
                            8,
                            8,
                        ),
                        type='CLAHE'),
                    dict(
                        alpha=(
                            0.2,
                            0.5,
                        ),
                        lightness=(
                            0.8,
                            1.2,
                        ),
                        p=0.5,
                        type='Sharpen'),
                    dict(blur_limit=(
                        3,
                        5,
                    ), p=0.2, type='GaussianBlur'),
                    dict(p=0.3, type='GaussNoise', var_limit=(
                        5,
                        15,
                    )),
                ],
                type='Albu',
                update_pad_shape=False),
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
    dict(
        direction='horizontal',
        label_swap_map=dict({
            4: 5,
            6: 7,
            8: 9
        }),
        prob=0.5,
        type='RandomFlipWithLabelSwap'),
    dict(degree=15, pad_val=0, prob=0.5, seg_pad_val=0, type='RandomRotate'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.8,
            1.1,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.7, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(
        keymap=dict(gt_semantic_seg='mask', img='image'),
        transforms=[
            dict(clip_limit=2.0, p=0.6, tile_grid_size=(
                8,
                8,
            ), type='CLAHE'),
            dict(
                alpha=(
                    0.2,
                    0.5,
                ),
                lightness=(
                    0.8,
                    1.2,
                ),
                p=0.5,
                type='Sharpen'),
            dict(blur_limit=(
                3,
                5,
            ), p=0.2, type='GaussianBlur'),
            dict(p=0.3, type='GaussNoise', var_limit=(
                5,
                15,
            )),
        ],
        type='Albu',
        update_pad_shape=False),
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
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
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
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
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
work_dir = '/home/featurize/work/AI6126project1/out/segformer_mit-b5'
