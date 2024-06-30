# model settings

_base_ = './efficient-hrdecoder_fcn_hr18.py'

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),                                
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    hr_settings=dict(
        in_channels=sum([48, 96, 192, 384]),
    ),
)