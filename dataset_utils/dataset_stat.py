from os import path, listdir
import argparse
import yaml

import numpy as np
from tqdm import tqdm

def load_clouds(dataset_path, cfg):
    NAV = yaml.safe_load(open(path.join(cfg, 'navigation.yml'), 'r'))

    for dataset_name in NAV:
        for split in NAV[dataset_name]:
            if not NAV[dataset_name][split]:
                continue
            for seq in NAV[dataset_name][split]:
                dir_path = path.join(dataset_path, dataset_name, 'data', str(seq), 'clouds')
                print('Reading: %s' % dir_path)
                cloud_file_names = sorted(listdir(dir_path))
                for cloud_file_name in tqdm(cloud_file_names):
                    cloud_file_path = path.join(dir_path, cloud_file_name)
                    # x, y, z, i
                    if cloud_file_path.endswith('.bin'):
                        yield np.fromfile(cloud_file_path, dtype=np.float32).reshape(-1, 4)
                    elif cloud_file_path.endswith('.npy'):
                        yield np.load(cloud_file_path)
                    else:
                        print('Error extension: %s' % cloud_file_path)
                        continue

def load_labels(dataset_path, cfg):
    NAV = yaml.safe_load(open(path.join(cfg, 'navigation.yml'), 'r'))

    for dataset_name in NAV:
        for split in NAV[dataset_name]:
            if not NAV[dataset_name][split]:
                continue
            for seq in NAV[dataset_name][split]:
                dir_path = path.join(dataset_path, dataset_name, 'data', str(seq), 'labels')
                print('Reading: %s' % dir_path)
                label_file_names = sorted(listdir(dir_path))
                for label_file_name in tqdm(label_file_names):
                    label_file_path = path.join(dir_path, label_file_name)
                    # load label
                    if label_file_path.endswith('.label'):
                        yield np.fromfile(label_file_path, dtype=np.int32).reshape((-1))
                    elif label_file_path.endswith('.npy'):
                        yield np.load(label_file_path, allow_pickle=True)
                    else:
                        print('Error extension: %s' % label_file_path)
                        continue

def calculate_mean_stats(cloud):
    range_sum = np.sum(np.linalg.norm(cloud[:, :3], axis=1))
    coord_sums = np.sum(cloud, axis=0)
    num_points = np.array([len(cloud)])
    stats = np.hstack([range_sum, coord_sums, num_points]).astype(np.float32)
    return stats

def calculate_std_stats(cloud, means):
    range_sum = np.sum((np.linalg.norm(cloud[:, :3], axis=1) - means[0]) ** 2)
    coord_sums = np.sum((cloud - means[1:5]) ** 2, axis=0)
    num_points = np.array([len(cloud)])
    stats = np.hstack([range_sum, coord_sums, num_points]).astype(np.float32)
    return stats

def calculate_stats(dataset_path, cfg):
    # calc means
    # range, x, y, z, i, count
    means = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    print('Calculating means...')
    num_clouds = 0
    for cloud in load_clouds(dataset_path, cfg):
        means += calculate_mean_stats(cloud)
        num_clouds += 1
    means[:5] /= means[5]

    # calc stds
    # range, x, y, z, i, count
    stds = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    print('Calculating stds...')
    for cloud in load_clouds(dataset_path, cfg):
        stds += calculate_std_stats(cloud, means)
    ddof = 1
    stds[:5] = np.sqrt(stds[:5] / (stds[5] - ddof))

    # calc content
    labl_count = {}
    print('Calculating content...')
    for label in load_labels(dataset_path, cfg):
        lbls, cnt = np.unique(label, return_counts=True)
        for i, lbl in enumerate(lbls):
            if lbl not in labl_count:
                labl_count[lbl] = cnt[i]
            else:
                labl_count[lbl] += cnt[i]

    labl_count = {k: labl_count[k] for k in sorted(labl_count.keys())}
    print(labl_count)
    total_count = sum(labl_count.values())

    # showing results
    print('PC count: %s' % num_clouds)
    keys = ['r', 'x', 'y', 'z', 'i']
    print('img_means:')
    for key, mean in zip(keys, means[:5]):
        print('\t- %.2f' % (mean))
    print('img_stds:')
    for key, std in zip(keys, stds[:5]):
        print('\t- %.2f' % (std))
    print('content:')
    for key in labl_count.keys():
        print('\t%s: %.16f' % (key, labl_count[key]/total_count))

# FIXME: need IMAGE stats, not clouds!
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Calculate point cloud stats from dataset dir with cloud files')
    argparser.add_argument('dataset_path', help='path to dataset directory')
    argparser.add_argument('cfg', help='path to configs directory')
    args = argparser.parse_args()
    calculate_stats(args.dataset_path, args.cfg)

# python dataset_stat.py /Dataset/ ../cfgs/
