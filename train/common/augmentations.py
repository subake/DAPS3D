#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time
from xml.etree.ElementTree import TreeBuilder

import numpy as np
import torch
import glob
import os

from PIL import Image

from scipy.stats import mode

class PointCloudAugmentator(object):
    def __init__(self, AUGMENT=None):
        self.augment = AUGMENT
        self.augmentations_fov = []
        self.augmentations_pc = []
        self.augmentations_labels = []
        self.augmentations_proj = []

        if self.augment['augment_fov']['use_augment_fov']:
            self.augmentations_fov.append(self.fov_augment)

        if self.augment['human_augmentation']['use_human_augmentation']:
            self.augmentations_pc.append(self.human_augmentation)
            self.augmentations_labels.append(self.human_augmentation_label)

        if self.augment['z_jitter']['use_z_jitter']:
            self.augmentations_pc.append(self.z_jitter) 

        if self.augment['xy_flip']['use_xy_flip']:
            self.augmentations_pc.append(self.xy_flip) 

        if self.augment['xy_rotate']['use_xy_rotate']:
            self.augmentations_pc.append(self.xy_rotate)

        if self.augment['pixel_dropout']['use_pixel_dropout']:
            self.augmentations_proj.append(self.pixel_dropout)

        if self.augment['tzone_mask']['use_tzone_mask']:
            self.augmentations_proj.append(self.tzone_aug_proj)

        self.augmentations_pc.append(self.zero_depth_fix_pc)
        self.augmentations_labels.append(self.zero_depth_fix_labels)

        augment_folder = '/mnt/hdd8/Datasets/nuscenes/augment_clouds'
        self.augment_files = glob.glob(os.path.join(augment_folder, '*.npy'))
        
        self.aug_prob = 0.5

    def augment_fov(self, fov, fov_up, fov_down):
        for augmentation in self.augmentations_fov:
            fov, fov_up, fov_down = augmentation(fov, fov_up, fov_down)

        return fov, fov_up, fov_down

    def augment_pc(self, point_cloud, remissions):
        for augmentation in self.augmentations_pc:
            point_cloud, remissions = augmentation(point_cloud, remissions)

        return point_cloud, remissions

    def augment_labels(self, labels):
        for augmentation in self.augmentations_labels:
            labels = augmentation(labels)

        return labels

    def augment_proj(self, proj_range, proj_xyz, proj_remission, proj_idx):
        for augmentation in self.augmentations_proj:
            proj_range, proj_xyz, proj_remission, proj_idx = augmentation(proj_range, proj_xyz, proj_remission, proj_idx)

        return proj_range, proj_xyz, proj_remission, proj_idx

    def human_augmentation(self, point_cloud, remissions):
        self.augment_human_labels = np.zeros((0), dtype=np.int32)

        # get depth of all points
        depth = np.linalg.norm(point_cloud, 2, axis=1)

        # PATCH for zero values
        depth += 1e-6

        depth_threshold = self.augment['earth_depth_threshold']
        max_aug_clouds = self.augment['human_augmentation']['max_human_augment']
        down_z, up_z = self.augment['human_augmentation']['human_z_low'], self.augment['human_augmentation']['human_z_high']

        # Augment people insertion
        close_points = point_cloud[depth < depth_threshold]
        ground_z = mode(close_points[:, 2])[0][0]

        # ground_z = range
        cloud_amount = np.random.randint(0, max_aug_clouds)

        for i in range(cloud_amount):
            alpha = np.random.rand() * 2 * np.pi

            id = np.random.randint(0, len(self.augment_files))
            cloud = np.load(self.augment_files[id])

            cloud[:, 2] += (ground_z - np.min(cloud[:, 2]))

            rotmat = np.array([
                [np.cos(alpha), np.sin(alpha)],
                [-np.sin(alpha), np.cos(alpha)]
            ])

            cloud[:, :2] = cloud[:, :2] @ rotmat
            cloud[:, 2] += np.random.uniform(down_z, up_z)

            point_cloud = np.concatenate([point_cloud, cloud])
            remissions = np.concatenate([remissions, np.zeros(len(cloud))])
            self.augment_human_labels = np.concatenate([self.augment_human_labels, np.full(len(cloud), 30, dtype=np.int32)])

        return point_cloud, remissions

    def human_augmentation_label(self, labels):
        labels = np.concatenate([labels, self.augment_human_labels])

        return labels


    def z_jitter(self, point_cloud, remissions):
        # get depth of all points
        depth = np.linalg.norm(point_cloud, 2, axis=1)

        # PATCH for zero values
        depth += 1e-6

        depth_threshold = self.augment['earth_depth_threshold']
        down_z, up_z = self.augment['z_jitter']['ego_z_low'], self.augment['z_jitter']['ego_z_high']

        close_points = point_cloud[depth < depth_threshold]
        ground_z = mode(close_points[:, 2])[0][0]

        # Augment z-jitter
        if np.random.rand() < self.aug_prob:
            point_cloud[:, 2] += np.random.uniform(-ground_z + down_z, -ground_z + up_z)

        return point_cloud, remissions

    def xy_flip(self, point_cloud, remissions):
        # Augment xy-flip
        if np.random.rand() < self.aug_prob:
            point_cloud[:, 0] = -point_cloud[:, 0]

        return point_cloud, remissions

    def xy_rotate(self, point_cloud, remissions):
        # Augmet xy-rotate
        if np.random.rand() < self.aug_prob:
            alpha = np.random.rand() * 2 * np.pi
            rotmat = np.array([
                [np.cos(alpha), np.sin(alpha)],
                [-np.sin(alpha), np.cos(alpha)]
            ])
            point_cloud[:, :2] = point_cloud[:, :2] @ rotmat

        return point_cloud, remissions


    def fov_augment(self, fov, fov_up, fov_down):
        # Augment FoV
        # alpha corresponds to possible masked area 
        alpha = self.augment['augment_fov']['alpha']
        fov_max = self.augment['augment_fov']['fov_max']

        fov = np.random.uniform(fov, fov_max)
        fov_down = np.random.uniform(fov_up - (1 + alpha) * fov, fov_down + alpha * fov)
        fov_up = fov_down + fov

        return fov, fov_up, fov_down

    def pixel_dropout(self, proj_range, proj_xyz, proj_remission, proj_idx):
        dropout_count = self.augment['pixel_dropout']['dropout_count']
        x_alpha, x_beta = self.augment['pixel_dropout']['x_alpha'], self.augment['pixel_dropout']['x_beta']
        y_alpha, y_beta = self.augment['pixel_dropout']['y_alpha'], self.augment['pixel_dropout']['y_beta']

        for i in range(dropout_count):
            x_win_left = np.random.randint(0, proj_range.shape[1])
            x_win_size = int(np.random.beta(x_alpha, x_beta) * proj_range.shape[1])
            x_win_right = x_win_left + x_win_size

            y_win_size = int(np.random.beta(y_alpha, y_beta) * proj_range.shape[0])
            y_win_down = np.random.randint(0, proj_range.shape[0] - y_win_size + 1)
            y_win_up = y_win_down + y_win_size

            if x_win_right > proj_range.shape[1]:
                # Right part
                proj_range[y_win_down:y_win_up, x_win_left:] = 0
                proj_xyz[y_win_down:y_win_up, x_win_left:] = 0
                proj_remission[y_win_down:y_win_up, x_win_left:] = 0
                proj_idx[y_win_down:y_win_up, x_win_left:] = 0

                # Left part
                proj_range[y_win_down:y_win_up, :x_win_right - x_win_left] = 0
                proj_xyz[y_win_down:y_win_up, :x_win_right - x_win_left] = 0
                proj_remission[y_win_down:y_win_up, :x_win_right - x_win_left] = 0
                proj_idx[y_win_down:y_win_up, :x_win_right - x_win_left] = 0
            else:
                proj_range[y_win_down:y_win_up, x_win_left:x_win_right] = 0
                proj_xyz[y_win_down:y_win_up, x_win_left:x_win_right] = 0
                proj_remission[y_win_down:y_win_up, x_win_left:x_win_right] = 0
                proj_idx[y_win_down:y_win_up, x_win_left:x_win_right] = 0

        return proj_range, proj_xyz, proj_remission, proj_idx

    def mask_noise(self, proj_range, proj_xyz, proj_remission, proj_idx):
        window_count = self.augment['mask_noise']['window_count']
        x_alpha, x_beta = self.augment['mask_noise']['x_alpha'], self.augment['mask_noise']['x_beta']
        y_alpha, y_beta = self.augment['mask_noise']['y_alpha'], self.augment['mask_noise']['y_beta']

        scale = self.augment['mask_noise']['scale']

        for i in range(window_count):
            x_win_left = np.random.randint(0, proj_range.shape[1])
            x_win_size = int(np.random.beta(x_alpha, x_beta) * proj_range.shape[1])
            x_win_right = x_win_left + x_win_size

            y_win_size = int(np.random.beta(y_alpha, y_beta) * proj_range.shape[0])
            y_win_down = np.random.randint(0, proj_range.shape[0] - y_win_size + 1)
            y_win_up = y_win_down + y_win_size

            denoise_mask = (proj_idx == 0)

            if x_win_right > proj_range.shape[1]:
                # Right part
                denoise_mask[y_win_down:y_win_up, x_win_left:] = 0
                denoise_mask[y_win_down:y_win_up, :x_win_right - x_win_left] = 0
            else:
                denoise_mask[y_win_down:y_win_up, x_win_left:x_win_right] = 0

            # Do not update range cause no need
            proj_xyz += (1 - denoise_mask) * np.random.normal(loc=0, scale=scale, size=proj_xyz.shape)

        return proj_range, proj_xyz, proj_remission, proj_idx

    def tzone_aug_proj(self, proj_range, proj_xyz, proj_remission, proj_idx):
        mask_path = self.augment['tzone_mask']['mask_path']
        t_mask = (np.array(Image.open(mask_path)) == 1)

        proj_range[t_mask] = 0
        proj_xyz[t_mask] = 0
        proj_remission[t_mask] = 0
        self.tzone_idx = proj_idx[t_mask]

        return proj_range, proj_xyz, proj_remission, proj_idx

    def zero_depth_fix_pc(self, point_cloud, remissions):
        self.zero_point = (point_cloud**2).sum(axis=1) == 0

        return point_cloud, remissions

    def zero_depth_fix_labels(self, labels):
        labels[self.zero_point] = 0

        return labels
