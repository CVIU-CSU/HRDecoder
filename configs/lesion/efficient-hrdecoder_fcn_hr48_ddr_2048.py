_base_ = [
    '../_base_/models/efficient-hrdecoder_fcn_hr48.py',
    '../_base_/datasets/hr_ddr_2048.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]

model = dict(
    type='EfficientHRDecoder',
    use_sigmoid=True,
    hr_settings=dict(
        visual_dim = 256,
        hr_scale = (1024,1024),
        scale_ratio = (0.75, 1.25),
        divisible = 8,
        lr_loss_weight = 0,
        hr_loss_weight = 0.1,
        fuse_mode = 'simple',
        crop_num = 4,
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=64,
        kernel_size=7,
        num_convs=1,
        compress=True,
        concat_input=False,
        num_classes=4,
        align_corners=False,
        loss_decode=dict(
            type='BinaryLoss', loss_type='dice', loss_weight=1.0, smooth=1e-5)
        ),# model training and testing settings
)

data=dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)
seed = 2
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=4000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')
