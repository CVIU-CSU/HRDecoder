norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='HRDecoder',
    use_sigmoid=True,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    hr_settings=dict(
        hr_scale=(960, 960),
        scale_ratio=(0.75, 1.25),
        divisible=8,
        lr_loss_weight=0,
        hr_loss_weight=0.1,
        fuse_mode='simple',
        crop_num=2),
    decode_head=dict(
        type='FCNHead',
        in_channels=720,
        in_index=0,
        channels=64,
        kernel_size=7,
        num_convs=1,
        compress=True,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=4,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='BinaryLoss', loss_type='dice', loss_weight=1.0,
            smooth=1e-05)),
    train_cfg=dict(
        work_dir='./work_dirs/hrdecoder_fcn_hr48_idrid_2880x1920-slide'),
    test_cfg=dict(
        mode='slide',
        compute_aupr=True,
        stride=(960, 960),
        crop_size=(1920, 1920)))
dataset_type = 'LesionDataset'
data_root = './data/IDRID'
img_norm_cfg = dict(
    mean=[116.513, 56.437, 16.309], std=[80.206, 41.232, 13.293], to_rgb=True)
image_scale = (2880, 1920)
crop_size = (1920, 1920)
palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]
classes = ['bg', 'EX', 'HE', 'SE', 'MA']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2880, 1920), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(1920, 1920), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(
        type='RandomRotate',
        prob=1.0,
        pad_val=0,
        seg_pad_val=0,
        degree=(-45, 45),
        auto_bound=False),
    dict(
        type='Normalize',
        mean=[116.513, 56.437, 16.309],
        std=[80.206, 41.232, 13.293],
        to_rgb=True),
    dict(type='Pad', size=(1920, 1920), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2880, 1920),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[116.513, 56.437, 16.309],
                std=[80.206, 41.232, 13.293],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_dir='images/train',
        ann_dir='labels/train',
        data_root='./data/IDRID',
        classes=['bg', 'EX', 'HE', 'SE', 'MA'],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128]],
        type='LesionDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2880, 1920), ratio_range=(0.5, 2.0)),
            dict(
                type='RandomCrop', crop_size=(1920, 1920), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(
                type='RandomRotate',
                prob=1.0,
                pad_val=0,
                seg_pad_val=0,
                degree=(-45, 45),
                auto_bound=False),
            dict(
                type='Normalize',
                mean=[116.513, 56.437, 16.309],
                std=[80.206, 41.232, 13.293],
                to_rgb=True),
            dict(type='Pad', size=(1920, 1920), pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        img_dir='images/test',
        ann_dir='labels/test',
        data_root='./data/IDRID',
        classes=['bg', 'EX', 'HE', 'SE', 'MA'],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128]],
        type='LesionDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2880, 1920),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[116.513, 56.437, 16.309],
                        std=[80.206, 41.232, 13.293],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        img_dir='images/test',
        ann_dir='labels/test',
        data_root='./data/IDRID',
        classes=['bg', 'EX', 'HE', 'SE', 'MA'],
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128]],
        type='LesionDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2880, 1920),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[116.513, 56.437, 16.309],
                        std=[80.206, 41.232, 13.293],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
seed = 2
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(
    interval=2000, metric='mIoU', priority='LOW', save_best='mIoU')
work_dir = './work_dirs/hrdecoder_fcn_hr48_idrid_2880x1920-slide'
gpu_ids = range(0, 1)
