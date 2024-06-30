_base_ = [
    '../_base_/models/efficient-hrdecoder_fcn_hr48.py',
    '../_base_/datasets/hr_idrid_2880x1920-slide.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]

model = dict(
    type='EfficientHRDecoder',
    use_sigmoid=True,
    hr_settings=dict(
        hr_scale = (960,960),
        scale_ratio = (0.75, 1.25),
        divisible = 8,
        lr_loss_weight = 0,
        hr_loss_weight = 0.1,
        fuse_mode = 'simple',
        crop_num = 2,
    ),
    test_cfg=dict(
        mode='slide',
        stride=(960,960),
        crop_size=(1920,1920),
        compute_aupr=True,
    )
)

data=dict(
    samples_per_gpu=1, 
    workers_per_gpu=1,
)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=4000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')
