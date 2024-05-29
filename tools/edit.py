import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

import argparse
import multiprocessing as mp
import platform
import warnings

import cv2
import mmcv
import torch
import json
import numpy as np
import pickle
from pytorch3d import ops
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmgen.apis import set_random_seed
from mmgen.core import build_metric
from mmgen.datasets import build_dataset
from mmgen.models import build_model
from mmgen.utils import get_root_logger
from lib.apis import evaluate_3d
from lib.datasets import build_dataloader
from lib.models.losses import PerLoss

_distributed_metrics = ['FID', 'IS', 'FIDKID']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test and eval a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

def load_smpl(path, smpl_type='smplx'):
    filetype = path.split('.')[-1]
    with open(path, 'rb') as f:
        if filetype=='pkl':
            smpl_param_data = pickle.load(f)
        elif filetype == 'json':
            smpl_param_data = json.load(f)
        else:
            assert False

    with open(os.path.join(os.path.split(path)[0][:-5], 'pose', '000_000.json'), 'rb') as f:
        tf_param = json.load(f)
    
    if smpl_type=='smpl':
        smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array(tf_param['center'])[None], 
                    smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    elif smpl_type == 'smplx':
        # for custom
        # smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array(tf_param['center']).reshape(1, -1), 
        #             np.zeros_like(np.array(smpl_param_data['global_orient']).reshape(1, -1)), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
        #             np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
        #             np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
        #             np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)

        # for thuman
        smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array([0, 0.35, 0]).reshape(1, -1), 
                    np.array(smpl_param_data['global_orient']).reshape(1, -1), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
                    np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
                    np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
                    np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)
    else:
        assert False
    # smpl_param = np.concatenate([smpl_param_data['scale'][:, None], smpl_param_data['transl'], 
    #                 smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    return torch.from_numpy(smpl_param.astype(np.float32)).reshape(-1)

def load_pose(path):
    with open(path, 'rb') as f:
        pose_param = json.load(f)
    w2c = np.array(pose_param['cam_param'], dtype=np.float32).reshape(36,4,4)
    cam_center = w2c[:, :3, 3]
    c2w = np.linalg.inv(w2c)
    # pose[:,:2] *= -1
    # pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(9, 4)
    c2w = torch.from_numpy(c2w)
    cam_to_ndc = torch.cat([c2w[:, :3, :3], c2w[:, :3, 3:]], dim=-1)
    pose = torch.cat([cam_to_ndc, cam_to_ndc.new_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(36, -1, -1)], dim=-2)

    return [pose, torch.from_numpy(cam_center)]

def tv_loss(tensor, dims=[-2, -1], power=1.5):
    shape = list(tensor.size())
    diffs = []
    for dim in dims:
        pad_shape = deepcopy(shape)
        pad_shape[dim] = 1
        diffs.append(torch.cat([torch.diff(tensor, dim=dim), tensor.new_zeros(pad_shape)], dim=dim))
    # return torch.stack(diffs, dim=0).norm(dim=0).pow(power).mean(dim=dims)
    return torch.stack(diffs, dim=0).norm(dim=0).pow(power).mean()

def reg_loss(tensor, power=2):
    return tensor.abs().mean() if power == 1 \
        else (tensor.abs() ** power).mean()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    dirname = os.path.dirname(args.checkpoint)
    ckpt = os.path.basename(args.checkpoint)

    if 'http' in args.checkpoint:
        log_path = None
    else:
        log_name = ckpt.split('.')[0] + '_eval_log' + '.txt'
        log_path = os.path.join(dirname, log_name)

    logger = get_root_logger(
        log_file=log_path, log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    model.eval()

    for eval_cfg in cfg.evaluation:
        if args.data is not None:
            if eval_cfg.data not in args.data:
                continue

        # The default loader config
        loader_cfg = dict(
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.get('val_workers_per_gpu',
                                         cfg.data.workers_per_gpu),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader', 'val_uncond', 'val_cond'
            ]
        })

        # load points data
        pcd_map = torch.as_tensor(np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/init_posmap_smplx_thu.npy')).cuda() # 1, 3, 256, 256
        pcd_map_re = pcd_map.permute(0, 2, 3, 1).reshape(1, -1, 3).cuda()
        # load_data_path
        data_latent_code = torch.load('/home/zhangweitian/HighResAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_thuman_conv_final/viz_uncond_code/scene_0225.pth')
        # data_latent_code = torch.load('/home/zhangweitian/HighResAvatar/work_dirs/ssdnerf_avatar_uncond_16bit_thuman_conv_final/viz_uncond_code/scene_0002.pth')
        geo_code = torch.nn.parameter.Parameter(data_latent_code[:16], requires_grad=False)
        tex_code = torch.nn.parameter.Parameter(data_latent_code[16:], requires_grad=True)
        code = [geo_code.unsqueeze(0).cuda(), tex_code.unsqueeze(0).cuda()]
        test_cam_path = '/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_train/0021/pose/340_001.json'
        test_smpl_path = '/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_train/0021/smplx/smplx_param.pkl'
        target_image_path = '/home/zhangweitian/HighResAvatar/edit/tar6.png'
        # test_cam_path = '/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_train/0005/pose/160_001.json'
        # test_smpl_path = '/home/zhangweitian/HighResAvatar/data/humanscan_wbg/human_train/0005/smplx/smplx_param.pkl'
        # target_image_path = '/home/zhangweitian/HighResAvatar/edit/tar5.png'
        # load pose data
        with open(test_cam_path, 'rb') as f:
            pose_param = json.load(f)
        cam_matrix = np.array(pose_param['cam_param'], dtype=np.float32).reshape(4,4)
        cam_center = cam_matrix[:3, 3]
        cam_matrix_inv = np.linalg.inv(cam_matrix)
        cam = torch.FloatTensor(torch.from_numpy(cam_matrix_inv))
        test_intrinsics = torch.FloatTensor(torch.from_numpy(cam_center)).unsqueeze(0).unsqueeze(1).cuda()
        cam_to_ndc = cam[:3]
        test_poses = torch.cat([cam_to_ndc, cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=-2).unsqueeze(0).unsqueeze(1).cuda()
        # load smpl data
        test_smpl_param = load_smpl(test_smpl_path).unsqueeze(0).cuda()
        # load image data
        img = mmcv.imread(target_image_path, channel_order='rgb')
        img = Image.open(target_image_path)
        img = np.asarray(img.resize((1024,1024), resample=Image.Resampling.LANCZOS))[:,:,:3]
        target_image = torch.from_numpy(img.astype(np.float32) / 255).unsqueeze(0).unsqueeze(1).cuda()
        # plt.imsave(os.path.join('/home/zhangweitian/HighResAvatar/edit_viz', 'target.jpg'), torch.round(target_image[0,0] * 255).to(torch.uint8).cpu().numpy())
        # assert False
        # training setup
        # optimizer = torch.optim.Adam([tex_code,], lr=0.02)
        # optimizer = torch.optim.Adam([tex_code,], lr=1e-2)
        optimizer = torch.optim.Adam([tex_code,], lr=5e-3)
        decoder = model.module.decoder_ema
        # decoder = model.decoder
        cfg = model.module.test_cfg
        h, w = cfg['img_size']
        l2loss = torch.nn.MSELoss()
        perloss = PerLoss(loss_weight=0.01).cuda()
        # begin to train
        # test_intrinsics [1, 1, 3]
        # test_poses [1, 1, 5, 4]
        # test_smpl_param [1, 123]
        # code [1, 32, 256, 256]
        # target_image [1, 1024, 1024, 3]
        with torch.no_grad():
            _, _, viz_mask = model.module.render(
                decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg, mask=None, return_norm=False, return_viz=True)
            viz_mask = viz_mask[:, 0].unsqueeze(0)
            visible_points = decoder.init_pcd[viz_mask]
            dist_sq, idx, neighbors = ops.knn_points(pcd_map_re, visible_points.unsqueeze(0), K=1, return_nn=True)
            viz_uv = (dist_sq < 0.0001)[0]
            # viz_uv = viz_uv.unsqueeze(0).reshape(1, 256, 256, -1)
            viz_uv = viz_uv.reshape(256, 256)
            # viz_uv_re = (viz_uv * 255).to(torch.uint8).cpu().numpy()
            # plt.imsave(os.path.join('/home/zhangweitian/HighResAvatar/edit_viz', 'viz_uv_2.jpg'), viz_uv_re)
            # assert False

        progress_bar = tqdm(range(100), desc="Training progress")
        for i in range(100):
            optimizer.zero_grad()
            image, _ = model.module.render(
                decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg, mask=viz_mask, return_norm=False)
            loss = l2loss(image, target_image) * 10
            per_loss = perloss(image.reshape(1, 1024, 1024, 3), target_image.reshape(1, 1024, 1024, 3))
            loss += per_loss
            tvloss = tv_loss(tex_code)
            loss += tvloss * 5e-3
            regloss = reg_loss(tex_code) * 3e-3
            loss += regloss
            loss.backward()
            tex_code.grad[:, ~viz_uv] = 0
            optimizer.step()
            with torch.no_grad():
                # if i % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(1)
            
        # save 360 degree rendering results
        with torch.no_grad():
            cam_poses = load_pose('/home/zhangweitian/HighResAvatar/data/cam_36.json')
            view_poses = cam_poses[0].unsqueeze(0).cuda()
            view_intrinsics = cam_poses[1].unsqueeze(0).cuda()
            view_images, _ = model.module.render(
                    decoder, code, h, w, view_intrinsics, view_poses, test_smpl_param, cfg=cfg, return_norm=False)
            
            pred_imgs = view_images.clamp(min=0, max=1).reshape(36, h, w, 3)
            output_viz = torch.round(pred_imgs * 255).to(torch.uint8).cpu().numpy()
            for i in range(36):
                plt.imsave(os.path.join('/home/zhangweitian/HighResAvatar/edit_viz', str(i)+'.jpg'), output_viz[i])


    return


if __name__ == '__main__':
    main()