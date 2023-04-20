import os
from os import path

import numpy as np

import torch
from torch.utils.data import Dataset

from common.laserscan import LaserScan, SemLaserScan
from common.augmentations import PointCloudAugmentator

EXTENSIONS_SCAN = ['.bin', '.npy']
EXTENSIONS_LABEL = ['.label', '.npy']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):
    def __init__(self, data_path, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, split):

        # save deats
        self.NAV = NAV
        self.LAB_PARAMS = LAB_PARAMS
        self.AUGMENT = AUGMENT
        self.split = split

        # saving stats
        self.sensor_img_means = torch.tensor(STATS['img_means'], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(STATS['img_stds'], dtype=torch.float)
        self.max_points = STATS['max_points']

        # does ground truth labels needed?
        self.gt = False
        if split in [ 'train', 'valid' ]:
            self.gt = True

        self.use_aug = False
        if split in ['train']:
            self.use_aug = True
        else:
            self.use_aug = False

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(LAB_PARAMS['learning_map_inv'])
        print('NUM CLASSES: %s' % self.nclasses)

        # sanity checks
        # make sure directory exists
        if os.path.isdir(data_path):
            print('Data folder exists: %s' % data_path)
        else:
            raise ValueError('Data folder doesn\'t exist! Exiting...')

        self.scan_file_paths = []
        self.label_file_paths = []
        self.params = {}

        # getting path to each scan and label
        for dataset_name in NAV.keys():
            # saving data params
            self.params[dataset_name] = SENSORS[dataset_name]

            if not NAV[dataset_name][split]:
                continue
            for seq in NAV[dataset_name][split]:
                # showing current sequence
                print('Parsing: %s -> %s -> %s' % (dataset_name, split, seq))

                # getting paths to scans
                scan_path = path.join(data_path, dataset_name, 'data', seq, 'clouds')
                for root, _, file_names in os.walk(os.path.expanduser(scan_path)):
                    for file_name in file_names:
                        if is_scan(file_name):
                            self.scan_file_paths.append({ 'path': os.path.join(root, file_name),
                                                          'dataset_name': dataset_name })
                self.scan_file_paths.sort(key=lambda d: d['path'])

                # getting paths to labels
                if self.gt:
                    label_path = path.join(data_path, dataset_name, 'data', seq, 'labels')

                    for root, _, file_names in os.walk(os.path.expanduser(label_path)):
                        for file_name in file_names:
                            if is_label(file_name):
                                self.label_file_paths.append({ 'path': os.path.join(root, file_name),
                                                               'dataset_name': dataset_name })
                    self.label_file_paths.sort(key=lambda d: d['path'])

                    assert(len(self.scan_file_paths) == len(self.label_file_paths))        

        for dataset_name, ps in self.params.items():
            print('%s\n%s' % (dataset_name, ps))

    def __getitem__(self, index):
        # getting path to scan file
        scan_file_desc = self.scan_file_paths[index]
        scan_file_path = scan_file_desc['path']

        # getting path to label file
        if self.gt:
            label_file_path = self.label_file_paths[index]['path']

        # getting data params
        params = self.params[scan_file_desc['dataset_name']]
        sensor_fov_up = params['sensor']['fov_up']
        sensor_fov_down = params['sensor']['fov_down']
        sensor_img_W = params['img_prop']['width']
        sensor_img_H = params['img_prop']['height']

        augmentator = PointCloudAugmentator(self.AUGMENT)

        # open a semantic laserscan
        if self.gt:
            scan = SemLaserScan(sem_color_dict=self.LAB_PARAMS['color_map'],
                                project=True,
                                H=sensor_img_H,
                                W=sensor_img_W,
                                fov_up=sensor_fov_up,
                                fov_down=sensor_fov_down,
                                max_classes=300,
                                use_aug=self.use_aug,
                                augmentator=augmentator)
        else:
            scan = LaserScan(project=True,
                             H=sensor_img_H,
                             W=sensor_img_W,
                             fov_up=sensor_fov_up,
                             fov_down=sensor_fov_down,
                             use_aug=self.use_aug,
                             augmentator=augmentator)

        # open and obtain scan
        proj_time = scan.open_scan(scan_file_path)
        if self.gt:
            scan.open_label(label_file_path)
            # map unused classes to used classes (also for projection)
            scan.sem_label = self.map(scan.sem_label, self.LAB_PARAMS['learning_map'])
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.LAB_PARAMS['learning_map'])
        
        # Step 1: making tensor of the uncompressed data (with the max num points)

        # determining actual number of points
        unproj_n_points = scan.points.shape[0]

        # filling tensor of shape max_num_points * (x, y, z)
        # with XYZ coords of actual points
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)

        # filling tensor of shape max_num_points
        # with range values of actual points
        unproj_range = torch.full((self.max_points,), -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)

        # the same for remission
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

        if self.gt:
            # the same for labels
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_idx = torch.from_numpy(scan.proj_idx).clone()
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        
        # WITH remission----------------------------------------
        # proj = torch.cat([proj_range.unsqueeze(0).clone(),
        #                                     proj_xyz.clone().permute(2, 0, 1),
        #                                     proj_remission.unsqueeze(0).clone()])
        # proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        # proj = proj * proj_mask.float()
        
        # WITHOUT remission-------------------------------------
        proj = torch.cat([
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
        ])
        proj = (proj - self.sensor_img_means[:4, None, None]) / self.sensor_img_stds[:4, None, None]
        proj = proj * proj_mask.float()
        # ------------------------------------------------------

        # get name and sequence
        path_norm = os.path.normpath(scan_file_path)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace('.bin', '')

        # return
        return ( proj,
                 proj_mask,
                 proj_labels,
                 unproj_labels,
                 path_seq,
                 path_name,
                 scan_file_desc['dataset_name'],
                #  proj_idx,
                 proj_x,
                 proj_y,
                 proj_range,
                 unproj_range,
                 proj_xyz,
                 unproj_xyz,
                 proj_remission,
                 unproj_remissions,
                 unproj_n_points,
                 proj_time )

    def __len__(self):
        return len(self.scan_file_paths)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print('Wrong key ', key)
        # do the mapping
        return lut[label]


class Parser():
    # standard conv, BN, relu
    def __init__(self, dataset_path, ARCH, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, batch_size):
        super(Parser, self).__init__()

        # saving params
        self.LAB_PARAMS = LAB_PARAMS

        # number of classes that matters is the one for xentropy
        self.nclasses = len(LAB_PARAMS['learning_map_inv'])

        # Train dataset
        self.train_dataset = SemanticKitti(dataset_path, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, split='train')
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=ARCH['train']['workers'],
                                                       drop_last=True)
        # assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

        # Valid dataset
        self.valid_dataset = SemanticKitti(dataset_path, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, split='valid')
        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=ARCH['train']['workers'],
                                                       drop_last=True)
        # assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

        # Test dataset
        self.test_dataset = SemanticKitti(dataset_path, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, split='test')
        self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=ARCH['train']['workers'],
                                                      drop_last=True)
        # assert len(self.testloader) > 0
        self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.LAB_PARAMS['labels'][idx]

    def get_xentropy_class_string(self, idx):
        return self.LAB_PARAMS['labels'][self.LAB_PARAMS['learning_map_inv'][idx]]

    def to_original(self, label):
        # put label in original values
        return SemanticKitti.map(label, self.LAB_PARAMS['learning_map_inv'])

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKitti.map(label, self.LAB_PARAMS['learning_map'])

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.LAB_PARAMS['learning_map_inv'])
        # put label in color
        return SemanticKitti.map(label, self.LAB_PARAMS['color_map'])
