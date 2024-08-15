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

    with open(os.path.join(os.path.split(path)[0], '000_000.json'), 'rb') as f:
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
        cloth_mask = torch.as_tensor(np.load('demo/transfer_exp/cloth_mask_thu.npy')).unsqueeze(0).cuda()
        pants_mask = torch.as_tensor(np.load('demo/transfer_exp/pants_mask_thu.npy')).unsqueeze(0).cuda()
        latent_code_a = torch.load('demo/transfer_exp/scene_0183.pth') # base person
        latent_code_b = torch.load('demo/transfer_exp/scene_0345.pth') # provide upper cloth
        latent_code_c = torch.load('demo/transfer_exp/scene_0371.pth') # provide pants
        decoder = model.module.decoder_ema
        pcd_map = torch.as_tensor(np.load('work_dirs/cache/init_posmap_smplx_thu.npy')).cuda() # 1, 3, 256, 256
        pcd_map_re = pcd_map.permute(0, 2, 3, 1).reshape(1, -1, 3).cuda()
        cloth_points = decoder.init_pcd[cloth_mask]
        dist_sq, idx, neighbors = ops.knn_points(pcd_map_re, cloth_points.unsqueeze(0), K=1, return_nn=True)
        cloth_uv = (dist_sq < 0.0001)[0]
        cloth_uv = cloth_uv.reshape(256, 256)
        cloth_uv[:15, :159] = False
        cloth_uv[151:, 155:] = False
        latent_code_a[:, cloth_uv] = latent_code_b[:, cloth_uv]
        # latent_code_a = latent_code_a.unsqueeze(0)
        # rows, cols = torch.nonzero(cloth_uv[80:, :], as_tuple=True)
        # cloth_uv_viz = (cloth_uv * 255).to(torch.uint8).cpu().numpy()
        # plt.imsave(os.path.join('demo/transfer_exp/result', 'cloth_uv_new.jpg'), cloth_uv_viz)


        pants_points = decoder.init_pcd[pants_mask]
        dist_sq, idx, neighbors = ops.knn_points(pcd_map_re, pants_points.unsqueeze(0), K=1, return_nn=True)
        pants_uv = (dist_sq < 0.0001)[0]
        pants_uv = pants_uv.reshape(256, 256)
        pants_uv[22:, :159] = False
        latent_code_a[:, pants_uv] = latent_code_c[:, pants_uv]
        latent_code_a = latent_code_a.unsqueeze(0)
        # pants_uv_viz = (pants_uv * 255).to(torch.uint8).cpu().numpy()
        # plt.imsave(os.path.join('demo/transfer_exp/result', 'pants_uv.jpg'), pants_uv_viz)


        test_smpl_path = 'demo/transfer_exp/smplx_param.pkl' # from scan 0223
        test_smpl_param = load_smpl(test_smpl_path).unsqueeze(0).cuda()
        cfg = model.module.test_cfg
        h, w = cfg['img_size']
            
        # # save 360 degree rendering results
        with torch.no_grad():
            cam_poses = load_pose('demo/cam_36.json')
            view_poses = cam_poses[0].unsqueeze(0).cuda()
            view_intrinsics = cam_poses[1].unsqueeze(0).cuda()
            view_images, _ = model.module.render(
                    decoder, latent_code_a, h, w, view_intrinsics, view_poses, test_smpl_param, cfg=cfg, return_norm=False)
            
            pred_imgs = view_images.clamp(min=0, max=1).reshape(36, h, w, 3)
            output_viz = torch.round(pred_imgs * 255).to(torch.uint8).cpu().numpy()
            os.makedirs('demo/transfer_exp/trans_viz', exist_ok=True)
            for i in range(36):
                plt.imsave(os.path.join('demo/transfer_exp/trans_viz', str(i)+'.jpg'), output_viz[i])

    return


if __name__ == '__main__':
    main()