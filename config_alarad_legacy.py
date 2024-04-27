from util import SELECTED_CONFIG, SELECTED_WEIGHT, DATA_DIR

# https://github.com/open-mmlab/mmdetection/blob/main/demo/MMDet_Tutorial.ipynb
# https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet
# Inherit and overwrite part of the config based on this config
_base_ = SELECTED_CONFIG
# _base_ = "./checkpoints/rtmdet_x_8xb32-300e_coco.py"
load_from = SELECTED_WEIGHT
data_root = DATA_DIR

# rtmdet x : max batch is 8, but for stability, I set 4

train_batch_size_per_gpu = 4
train_num_workers = 1   

max_epochs = 200
stage2_num_epochs = 1
base_lr = 0.00008


metainfo = {
    'classes': ('Bud', 'Flower', 'Receptacle',
                'Early_fruit', 'White_fruit',
                 '50%_maturity', '80%_maturity' ),
    'palette': [
        (220, 20, 60), (232,217,70), (135,204,68), (94,212,225), (62,63,246), (150,92,217), (242,160,43)
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/val/'),
        ann_file='val.json'))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/test/'),
        ann_file='test.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=f'{data_root}val.json')

test_evaluator = dict(ann_file=f'{data_root}test.json')

model = dict(bbox_head=dict(num_classes=len(metainfo['classes'])))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])