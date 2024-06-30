# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='HRDecoder',
    use_sigmoid=True,
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
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
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    hr_settings=dict(
        hr_scale=(1024,1024),
        scale_ratio=(0.75,1.25),
        divisible=8,
        lr_loss_weight=0,
        hr_loss_weight=0.1,
        fuse_mode = 'simple',
        crop_num = 2,
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=sum([18, 36, 72, 144]),
        in_index=0,
        channels=64,
        kernel_size=7,
        num_convs=1,
        compress=True,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='BinaryLoss', loss_type='dice', loss_weight=1.0, smooth=1e-5)
        ),# model training and testing settings
    train_cfg = dict(),
    test_cfg = dict(mode='whole',compute_aupr=True)
)
