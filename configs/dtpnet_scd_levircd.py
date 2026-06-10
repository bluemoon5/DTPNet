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
        out_indices=(2, 6, 11),     # fine-tuning
        num_tokens=50,              # fine-tuning
        drop_path_rate=0.0,
        frozen_exclude=['prompt_embeddings'],
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
        loss_decode=dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

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

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=1)