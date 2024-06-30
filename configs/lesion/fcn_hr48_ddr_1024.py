_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/hr_ddr_1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py'
]
model = dict(
    use_sigmoid=True,
    decode_head=dict(
        num_classes=4,
        loss_decode=dict(
            type='BinaryLoss', 
            loss_type='dice', 
            use_sigmoid=False, 
            loss_weight=1.0, 
            smooth=1e-5)
    )
)
data=dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)
lr=0.01
optimizer = dict(lr=lr, momentum=0.9, weight_decay=0.0005)
lr_config = dict(power=0.9, min_lr=lr/100)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')
