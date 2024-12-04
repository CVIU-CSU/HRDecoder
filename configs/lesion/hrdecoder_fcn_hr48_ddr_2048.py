_base_ = [
    '../_base_/models/hrdecoder_fcn_hr48.py',
    '../_base_/datasets/hr_ddr_2048.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]

model = dict(
    type= 'HRDecoder',
    hr_settings=dict(
        hr_scale = (1024,1024),
        scale_ratio = (0.75, 1.25),
        divisible = 8,
        lr_loss_weight = 0,
        hr_loss_weight = 0.1,
        fuse_mode = 'simple',
        crop_num = 4,
    ),
)

data=dict(
    samples_per_gpu=1, 
    workers_per_gpu=1,
)
seed = 14
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=4000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')
