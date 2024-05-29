import torch
import torch.nn as nn
import numpy as np
from mmgen.models.builder import MODULES
import lpips
import cv2


@MODULES.register_module()
class PerLoss(nn.Module):

    def __init__(self, loss_weight=1.0, height=1024, width=1024):
        super().__init__()
        self.loss_vgg = torch.jit.load('work_dirs/cache/vgg16.pt').eval()
        self.loss_weight = loss_weight
        self.height, self.width = height, width

    def forward(self, pred, target):
        rand_start = torch.randint(256, (1,))
        rand_start_h = ((torch.randn(1) * 256 / 3) + 256).int().clip(0, 512)
        if pred.shape[1] > 512:
            pred_imgs = pred.permute(0, 3, 1, 2).clamp(min=0, max=1)[:, :, rand_start[0]+20:rand_start[0]+532, rand_start_h[0]:rand_start_h[0]+512]
            target_imgs = target.permute(0, 3, 1, 2).clamp(min=0, max=1)[:, :, rand_start[0]+20:rand_start[0]+532, rand_start_h[0]:rand_start_h[0]+512]
        else:
            pred_imgs = pred.permute(0, 3, 1, 2).clamp(min=0, max=1)
            target_imgs = target.permute(0, 3, 1, 2).clamp(min=0, max=1)
        pred_imgs = pred_imgs.reshape(-1, 3, 512, 512)
        target_imgs = target_imgs.reshape(-1, 3, 512, 512)

        pred_imgs = pred_imgs * 255
        target_imgs = target_imgs * 255
        pred_feat = self.loss_vgg(pred_imgs, resize_images=False, return_lpips=True)
        target_feat = self.loss_vgg(target_imgs, resize_images=False, return_lpips=True)
        dist = (target_feat - pred_feat).square().sum()
        dist_weighted = dist * self.loss_weight
        return dist_weighted