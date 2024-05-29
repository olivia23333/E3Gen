from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import imageio
import json
import pickle
import time
import mmcv
from mmcv.runner import get_dist_info
from mmgen.core.evaluation.metrics import FID, IS
from mmgen.models.architectures.common import get_module_device


def evaluate_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    # viz_step = 1
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if metrics is not None and len(metrics) > 0:
            pred_imgs = outputs_dict['pred_imgs'][:,:,:3]
            pred_imgs = pred_imgs.reshape(
                -1, 3, *outputs_dict['pred_imgs'].shape[3:]).split(feed_batch_size, dim=0)
            real_imgs = None
            for metric in metrics:
                if 'test_imgs' in data and not isinstance(metric, (FID, IS)) and real_imgs is None:
                    real_imgs = data['test_imgs'].permute(0, 1, 4, 2, 3)[:,:,3]
                    print(real_imgs[0].shape)
                    assert False
                    real_imgs = real_imgs.reshape(-1, 3, *real_imgs.shape[3:]).split(feed_batch_size, dim=0)
                for batch_id, batch_imgs in enumerate(pred_imgs):
                    # feed in fake images
                    metric.feed(batch_imgs * 2 - 1, 'fakes')
                    if not isinstance(metric, (FID, IS)) and real_imgs is not None:
                        metric.feed(real_imgs[batch_id] * 2 - 1, 'reals')

        if rank == 0:
            pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars

def recon_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    # max_num_scenes = 64
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    viz_step=1
    # sampling fake images and directly send them to metrics
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if rank == 0:
            pbar.update(total_batch_size)
        if i == max_num_scenes//total_batch_size - 1:
            break

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars

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


def viz_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    viz_step = 1
    path = 'data/cam_36.json'
    cam_poses = load_pose(path)
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        data['cond_poses'] = cam_poses[0].unsqueeze(0).repeat(4, 1, 1, 1)
        data['cond_intrinsics'] = cam_poses[1].unsqueeze(0).repeat(4, 1, 1)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if rank == 0:
            pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars


def animate_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    viz_step = 1
    # path = '/home/zhangweitian/HighResAvatar/data/cam_36.json'
    # cam_poses = load_pose(path)
    # amass
    smplx_path = 'ani_file/CMU/06/06_13_stageii.npz'
    smplx_pose_param = np.load(smplx_path, allow_pickle=True)
    # with open(smplx_path, "rb") as file:
    #     smplx_pose_param = pickle.load(file)
    smplx_param = np.concatenate([smplx_pose_param['poses'][:, :3], smplx_pose_param['pose_body'], smplx_pose_param['pose_hand'], smplx_pose_param['pose_jaw'], smplx_pose_param['pose_eye']], axis=1)
    # smplx_param = np.concatenate([smplx_pose_param['global_orient'][:,0], smplx_pose_param['body_pose_axis'], smplx_pose_param['left_hand_pose'], smplx_pose_param['right_hand_pose'], smplx_pose_param['jaw_pose']+np.array([0.05, 0, 0]), smplx_pose_param['leye_pose'], smplx_pose_param['reye_pose']], axis=1)
    # smplx_param = torch.as_tensor(smplx_param[:500])
    smplx_param = torch.as_tensor(smplx_param[100:200])
    # pose_body, pose_hand, pose_jaw, pose_eye
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        data['cond_poses'] = data['cond_poses'][:, 31:32]
        data['cond_intrinsics'] = data['cond_intrinsics'][:, 31:32]
        betas = data['cond_smpl_param'][:, 70:80].unsqueeze(1).expand(-1, 50, -1).float()
        data['cond_smpl_param'] = torch.cat((smplx_param.unsqueeze(0).expand(4, -1, -1)[:, 0::2].to(betas.device).float(), betas), 2)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if rank == 0:
            pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars
