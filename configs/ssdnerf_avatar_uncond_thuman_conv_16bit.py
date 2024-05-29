name = 'ssdnerf_avatar_uncond_16bit_thuman_conv'

model = dict(
    type='DiffusionNeRF',
    code_size=(32, 256, 256),
    code_reshape=(32, 256, 256),
    code_activation=dict(
        type='NormalizedTanhCode', mean=0.0, std=0.5, clip_range=2),
    grid_size=64,
    diffusion=dict(
        type='GaussianDiffusion',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        denoising=dict(
            type='DenoisingUnetMod',
            image_size=256,
            in_channels=32,
            base_channels=128,
            channels_cfg=[0.5, 1, 2, 2, 4, 4],
            resblocks_per_downsample=2,
            dropout=0.0,
            use_scale_shift_norm=True,
            downsample_conv=True,
            upsample_conv=True,
            num_heads=4,
            attention_res=[32, 16, 8]),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=20)),
    decoder=dict(
        type='UVNDecoder',
        interp_mode='bilinear',
        base_layers=[16, 64],
        density_layers=[64, 1],
        color_layers=[16, 128, 9],
        offset_layers=[64, 3],
        use_dir_enc=False,
        dir_layers=[16, 64],
        activation='silu',
        bg_color=1,
        # sigma_activation='trunc_exp',
        sigma_activation='sigmoid',
        sigmoid_saturation=0.001,
        gender='male',
        max_steps=256,
        multires=0,
        image_size=1024,
        superres=False,
        ),
    decoder_use_ema=True,
    freeze_decoder=False,
    bg_color=1,
    pixel_loss=dict(
        type='MSELoss',
        loss_weight=20),
    per_loss=dict(
        type='PerLoss',
        loss_weight=0.005,
        height=512,
        width=512),
    cache_size=500,
    scale_loss_weight=0,
    cache_16bit=True)

save_interval = 5000
eval_interval = 5000
code_dir = 'cache/' + name + '/code'
work_dir = 'work_dirs/' + name

train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.005,
    extra_scene_step=3,
    n_inverse_rays=2 ** 12,
    n_decoder_rays=2 ** 12,
    loss_coef=0.1 / (1024 * 1024),
    optimizer=dict(type='Adam', lr=0.04, weight_decay=0.),
    cache_load_from=code_dir,
    viz_dir=None,
    densify_start_iter=20000,
    init_iter=500,
    offset_weight=50,
    scale_weight=0.01,)
test_cfg = dict(
    img_size=(1024, 1024),
    num_timesteps=50,
    clip_range=[-2, 2],
    density_thresh=0.005,
    # max_render_rays=16 * 128 * 128,
)

optimizer = dict(
    diffusion=dict(type='Adam', lr=1e-4, weight_decay=0.),
    decoder=dict(type='Adam', lr=1e-3, weight_decay=0.))
dataset_type = 'AvatarDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/humanscan_wbg/human_train',
        cache_path='data/humanscan_wbg/human_train_cache.pkl',
        specific_observation_num=16,
        img_res=1024),
    val_uncond=dict(
        type=dataset_type,
        data_prefix='data/humanscan_wbg/human_train',
        load_imgs=False,
        num_test_imgs=54,
        scene_id_as_name=True,
        img_res=1024,
        cache_path='data/humanscan_wbg/human_train_cache.pkl'),
    val_cond=dict(
        type=dataset_type,
        data_prefix='data/humanscan_wbg/human_test',
        specific_observation_idcs=[0],
        img_res=1024,
        cache_path='data/humanscan_wbg/human_test_cache.pkl'),
    train_dataloader=dict(split_data=True))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    gamma=0.5,
    step=[100000])
checkpoint_config = dict(interval=save_interval, by_epoch=False, max_keep_ckpts=2)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_uncond',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FIDKID',
            num_images=500 * 54,
            inception_pkl='work_dirs/cache/thuman_train_wbg_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),
        viz_dir=work_dir + '/viz_uncond',
        save_best_ckpt=False)]

total_iters = 200000
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', 'decoder_ema'),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=4, ema_rampup=0.05, batch_size=8, eps=1e-8),
        priority='VERY_HIGH'),
    dict(
        type='SaveCacheHook',
        interval=save_interval,
        by_epoch=False,
        out_dir=code_dir,
        viz_dir='cache/' + name + '/viz'),
    dict(
        type='ModelUpdaterHook',
        step=[2000, 50000, 100000],
        cfgs=[{'train_cfg.extra_scene_step': 1},
              {'train_cfg.extra_scene_step': 1,
               'train_cfg.offset_weight': 25,
               'pixel_loss.loss_weight': 10.0,
               'per_loss.loss_weight':0.0025,},
              {'train_cfg.extra_scene_step': 1,
               'train_cfg.optimizer.lr': 0.02,
               'train_cfg.offset_weight': 25,
               'pixel_loss.loss_weight': 10.0,
               'per_loss.loss_weight':0.0025,}],
        by_epoch=False)
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'