_base_ = [
    '../common/standard_512x512_40k_levircd.py']

crop_size = (512, 512)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[122.7709, 116.7460, 104.0937] * 2,
    std=[68.5005, 66.6322, 70.3232] * 2,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='DualSiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/dinov2_vitb14_pretrain.pth',
    asymetric_input=True,
    encoder_resolution=dict(
        size=(518, 518),
        mode='bilinear'),
    image_encoder=dict(
        type='DinoVisionTransformerV2',
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        out_indices=(2, 5, 10),     # fine-tuning
        num_tokens=128,              # fine-tuning
        drop_path_rate=0.0,
        frozen_exclude=['prompt_embeddings'],# prompt_embeddings
        prompted=True),
    decode_head=dict(
        type='DiffTransformer',
        difft_cfg=dict(
            embed_dims=768,
            num_heads=16,
            num_layers=3),
        ban_dec_cfg=dict(
            type='BAN_BITHead',
            in_channels=768,
            channels=32,
            num_classes=2),
        loss_decode=dict( type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    #decode_head=dict(
    #     type='BitemporalAdapterHead',
    #     ban_cfg=dict(
    #         clip_channels=768,
    #         fusion_index=[0, 1, 2],
    #         side_enc_cfg=dict(
    #             type='mmseg.ResNetV1c',
    #             init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'), # [] # type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'
    #             in_channels=3,
    #             depth=18,
    #             num_stages=3,
    #             out_indices=(2,),
    #             dilations=(1, 1, 1),
    #             strides=(1, 2, 1),
    #             norm_cfg=dict(type='SyncBN', requires_grad=True),
    #             norm_eval=False,
    #             style='pytorch',
    #             contract_dilation=True)),
    #     ban_dec_cfg=dict(
    #         type='BAN_BITHead',
    #         in_channels=256,
    #         channels=32,
    #         num_classes=2),
    #     loss_decode=dict( type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.),
            'mask_decoder': dict(lr_mult=10.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

# 分布式训练环境配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# 数据加载器配置
train_dataloader = dict(
    batch_size=4,  # 每个GPU的batch_size
    num_workers=4,  # 每个GPU的工作进程数
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False))

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))