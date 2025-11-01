from typing import Tuple
import os

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast

from ..builder import NECKS


@NECKS.register_module()
class FastrayTransformer(BaseModule):
    
    def __init__(
        self,
        grid_config,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        downsample: int = 1,
        stride: int = 8,
        fuse = None,
        use_depth = True,
        loss_depth_weight = 0.125,
        depth_act='sigmoid',
        sid = False,
        is_transpose = True,
        accelerate = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.grid_config = grid_config
        self.out_channels = out_channels
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()
        self.create_grid_infos(**grid_config)
        self.fuse = self.fuse_type = None
        if fuse is not None:
            if fuse['type'] == 's2c':
                self.fuse = nn.Conv2d(self.out_channels*self.grid_size[2].int(), self.out_channels, kernel_size=1)
            self.fuse_type = fuse['type']
        self.stride = stride
        self.is_transpose = is_transpose
        self.voxel_coords = self.create_voxel_coords()
        self.accelerate = accelerate
        assert depth_act in ['sigmoid', 'softmax']
        self.depth_act = depth_act
        self.sid = sid
        self.use_depth = use_depth
        self.loss_depth_weight = loss_depth_weight
        self.D = int((self.grid_config['depth'][1] - self.grid_config['depth'][0]) / self.grid_config['depth'][2])
        self.depth_net = nn.Conv2d(
            self.in_channels, self.D + self.out_channels, kernel_size=1, padding=0)

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_voxel_coords(self):
        x = torch.arange(int(self.grid_size[0])).view(-1, 1, 1).expand(-1, int(self.grid_size[1]), int(self.grid_size[2]))
        y = torch.arange(int(self.grid_size[1])).view(1, -1, 1).expand(int(self.grid_size[0]), -1, int(self.grid_size[2]))
        z = torch.arange(int(self.grid_size[2])).view(1, 1, -1).expand(int(self.grid_size[0]), int(self.grid_size[1]), -1)
        coords = torch.stack((x, y, z), dim=3)
        coords = coords * self.grid_interval + (self.grid_lower_bound - self.grid_interval / 2.0)
        coords = coords.reshape(-1, 3)
        return nn.Parameter(coords, requires_grad=False)

    def get_fastray_input(self, input):
        img, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = input[:7]
        batch_size, n_images, n_channels, height, width = img.shape
        lidar_aug_matrix = bda
        post_rots = post_rots
        post_trans = post_trans
        camera2lidar_rots = sensor2ego[..., :3, :3]
        camera2lidar_trans = sensor2ego[..., :3, 3]
        extra_rots = bda[..., :3, :3]
        extra_trans = bda[..., :3, 3]

        new_cam2imgs = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(*sensor2ego.shape[:2], 1, 1).to(sensor2ego.device)
        new_cam2imgs[:, :, :3, :3] = cam2imgs
        camego2imgs = new_cam2imgs.matmul(torch.inverse(sensor2ego))

        batch_pre_voxel_coors_list = []
        batch_pre_img_coors_list = []
        batch_pre_depth_coors_list = []
        for b in range(batch_size):
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_camego2img = camego2imgs[b]
            curr_post_rots = post_rots[b]
            curr_post_trans = post_trans[b]

            # inverse aug
            cur_coords = self.voxel_coords - cur_lidar_aug_matrix[:3, 3].view(1,3)
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))

            # camego2image
            cur_coords = cur_camego2img[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_camego2img[:, :3, 3].reshape(-1, 3, 1)

            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :][cur_coords[:, 2, :] <= 0.0] = torch.inf
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = curr_post_rots.matmul(cur_coords)
            cur_coords += curr_post_trans.reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]] / self.stride
            cur_coords = cur_coords.long()

            on_img = ((cur_coords[..., 0] < (self.image_size[0] / self.stride))
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < (self.image_size[1] / self.stride))
                    & (cur_coords[..., 1] >= 0)
                    & (dist >= self.grid_config['depth'][0])
                    & (dist < self.grid_config['depth'][1]))
            for valid_i in range(1, len(on_img)):
                for valid_j in range(0, valid_i):
                    on_img[valid_i][on_img[valid_j] == True] = False

            pre_img_coors_list = []
            pre_depth_coors_list = []
            pre_voxel_coors_list = []
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]]
                # assert masked_coords[(masked_coords[:, 0] == 0) & (masked_coords[:, 1] == 0)].shape[0] == 0
                pre_img_coors_list.append(torch.cat([
                    masked_coords.new(masked_coords[:, 0:1].shape).zero_()+c, 
                    masked_coords[:, 0:1], 
                    masked_coords[:, 1:2]
                ], dim=1))

                pre_voxel_coors_list.append(torch.nonzero(on_img[c])[:, 0])
                depth_idx = ((dist[c, on_img[c]] - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]).long() - 1
                if len(depth_idx) != 0:
                    assert depth_idx.min() >= 0
                    assert depth_idx.max() < self.D
                pre_depth_coors_list.append(depth_idx)

            pre_voxel_coors_index = torch.nonzero(~on_img.sum(0).bool())[:, 0]
            pre_img_coors_index = torch.zeros(pre_voxel_coors_index.shape[0], 3).long().to(pre_voxel_coors_index.device)
            pre_depth_coors_index = torch.zeros(pre_voxel_coors_index.shape[0]).long().to(pre_voxel_coors_index.device)
            pre_voxel_coors_list.append(pre_voxel_coors_index)
            pre_img_coors_list.append(pre_img_coors_index)
            pre_depth_coors_list.append(pre_depth_coors_index)

            pre_voxel_coors_list = torch.cat(pre_voxel_coors_list, dim=0)
            pre_img_coors_list = torch.cat(pre_img_coors_list, dim=0)
            pre_depth_coors_list = torch.cat(pre_depth_coors_list, dim=0)

            if self.accelerate:
                assert batch_size == 1, batch_size
                assert self.use_depth, self.use_depth
                new_img_indices = []
                new_depth_indices = []
                N = pre_img_coors_list.shape[0]
                for idx in range(N):
                    fc, fh, fw = pre_img_coors_list[idx]
                    fd = pre_depth_coors_list[idx]
                    new_img_indices.append(fc * (height * width) + fh * width + fw)
                    new_depth_indices.append((fc * (height * width) + fh * width + fw) * self.D + fd)
                pre_img_coors_list = torch.Tensor(new_img_indices).long().to(pre_img_coors_list.device)
                pre_depth_coors_list = torch.Tensor(new_depth_indices).long().to(pre_img_coors_list.device)

                # sort by voxel coor
                pre_voxel_coors_list, sort_idx = pre_voxel_coors_list.sort()
                pre_img_coors_list = pre_img_coors_list[sort_idx]
                pre_depth_coors_list = pre_depth_coors_list[sort_idx]
                assert len(pre_voxel_coors_list) == len(self.voxel_coords)

            batch_pre_voxel_coors_list.append(pre_voxel_coors_list)
            batch_pre_img_coors_list.append(pre_img_coors_list)
            batch_pre_depth_coors_list.append(pre_depth_coors_list)
        return batch_pre_voxel_coors_list, batch_pre_img_coors_list, batch_pre_depth_coors_list

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.stride,
                                   self.stride, W // self.stride,
                                   self.stride, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.stride * self.stride)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    torch.inf * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.stride,
                                   W // self.stride)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            raise NotImplemented
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, depth_from_lidar=None):
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)
        x = x.view(B, N, self.D + self.out_channels, H, W).permute(0, 1, 3, 4, 2)
        # warning: make sure not been sampled
        x[:, 0, 0, 0] = 0.0
        if self.depth_act == 'sigmoid':
            depth = x[..., :self.D].sigmoid()
        else:
            depth = x[..., :self.D].softmax(dim=-1)
        x = x[..., self.D:(self.D + self.out_channels)]
        pre_voxel_coors_list, pre_img_coors_list, pre_depth_coors_list = self.get_fastray_input(input)
        if self.accelerate:
            assert self.use_depth, self.use_depth
            x = x.reshape(-1, self.out_channels)
            depth = depth.reshape(-1)
            x = x[pre_img_coors_list[0]]
            depth = depth[pre_depth_coors_list[0]].unsqueeze(1)
            x = x * depth
            x = x.view(B, *self.grid_size.int().tolist(), self.out_channels)
        else:
            voxel_feature = torch.zeros(
                (B, int(self.grid_size[0]) * int(self.grid_size[1]) * int(self.grid_size[2]), self.out_channels), device=x.device
            ).type_as(x)
            for i in range(B):
                if self.use_depth:
                    voxel_feature[i][pre_voxel_coors_list[i]] = \
                        x[i][pre_img_coors_list[i][:, 0], pre_img_coors_list[i][:, 1], pre_img_coors_list[i][:, 2]] * \
                        depth[i][pre_img_coors_list[i][:, 0], pre_img_coors_list[i][:, 1], pre_img_coors_list[i][:, 2], pre_depth_coors_list[i]].unsqueeze(-1)
                else:
                    voxel_feature[i][pre_voxel_coors_list[i]] = \
                        x[i][pre_img_coors_list[i][:, 0], pre_img_coors_list[i][:, 1], pre_img_coors_list[i][:, 2]]
            x = voxel_feature.view(B, *self.grid_size.int().tolist(), self.out_channels)
            N, X, Y, Z, C = x.shape
        permute = [0, 3, 2, 1] if self.is_transpose else [0, 3, 1, 2]
        if self.fuse_type is not None:
            if self.fuse_type == 's2c':
                x = x.reshape(N, X, Y, Z*C).permute(permute)
                x = self.fuse(x)
            elif self.fuse_type == 'sum':
                x = x.sum(dim=-2).permute(permute)
            elif self.fuse_type == 'max':
                x = x.max(dim=-2)[0].permute(permute)
            else:
                raise NotImplemented
        x = self.downsample(x)
        return x, depth
    
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

