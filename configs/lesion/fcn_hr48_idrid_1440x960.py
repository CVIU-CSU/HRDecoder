_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/idrid_1440x960.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]
model = dict(
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
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, 
                  metric='mIoU',
                  priority='LOW',
                  save_best='mIoU')
