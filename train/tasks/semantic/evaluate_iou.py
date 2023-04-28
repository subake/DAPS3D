#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
from os import path
import yaml
import sys
import numpy as np
import torch

from tqdm import tqdm

from tasks.semantic.modules.ioueval import iouEval
from common.laserscan import SemLaserScan

EXTENSIONS_SCAN = ['.bin', '.npy']
EXTENSIONS_LABEL = ['.label', '.npy']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def save_to_log(logdir, logfile, message):
    f = open(logdir+'/'+logfile, 'a')
    f.write(message+'\n')
    f.close()
    return

def evaluate(data_path, NAV, STATS, SENSORS, LAB_PARAMS, split, pred_path):
    scan_file_paths = []
    label_file_paths = []
    pred_file_paths = []
    param_dict = {}

    for dataset_name in NAV.keys():
        # saving data params
        param_dict[dataset_name] = SENSORS[dataset_name]
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
                        scan_file_paths.append({ 'path': os.path.join(root, file_name),
                                                 'dataset_name': dataset_name })
            scan_file_paths.sort(key=lambda d: d['path'])

            # getting paths to labels
            label_path = path.join(data_path, dataset_name, 'data', seq, 'labels')
            for root, _, file_names in os.walk(os.path.expanduser(label_path)):
                for file_name in file_names:
                    if is_label(file_name):
                        label_file_paths.append({ 'path': os.path.join(root, file_name),
                                                    'dataset_name': dataset_name })
            label_file_paths.sort(key=lambda d: d['path'])

            # getting paths to preds
            dataset_pred_path = os.path.join(pred_path, dataset_name, seq)

            for root, _, file_names in os.walk(os.path.expanduser(dataset_pred_path)):
                print(root, len(file_names))
                for file_name in file_names:
                    if is_label(file_name):
                        pred_file_paths.append({ 'path': os.path.join(root, file_name),
                                                    'dataset_name': dataset_name })
            pred_file_paths.sort(key=lambda d: d['path'])

    # check that I have the same number of files
    print('scans: ', len(scan_file_paths))
    print('labels: ', len(label_file_paths))
    print('predictions: ', len(pred_file_paths))
    assert (len(label_file_paths) == len(scan_file_paths) and
            len(label_file_paths) == len(pred_file_paths))

    print(FLAGS.predictions)
    pred_log_file = 'iou.txt' 
    print('Evaluating sequences: ')
    # open each file, get the tensor, and make the iou comparison
    for scan_file_desc, label_file_desc, pred_file_desc in tqdm(list(zip(scan_file_paths, label_file_paths, pred_file_paths))[:]):
        # getting paths
        scan_file_path = scan_file_desc['path']
        label_file_path = label_file_desc['path']
        pred_file_path = pred_file_desc['path']

        # getting data params
        params = param_dict[scan_file_desc['dataset_name']]
        sensor_fov_up = params['sensor']['fov_up']
        sensor_fov_down = params['sensor']['fov_down']
        sensor_img_W = params['img_prop']['width']
        sensor_img_H = params['img_prop']['height']
        sensor_img_means = torch.tensor(STATS['img_means'], dtype=torch.float)
        sensor_img_stds = torch.tensor(STATS['img_stds'], dtype=torch.float)
        max_points = STATS['max_points']

        # open label
        label = SemLaserScan(sem_color_dict=LAB_PARAMS['color_map'],
                             project=False,
                             H=sensor_img_H,
                             W=sensor_img_W,
                             fov_up=sensor_fov_up,
                             fov_down=sensor_fov_down,
                             max_classes=300)
        label.open_scan(scan_file_path)
        label.open_label(label_file_path)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_label_sem = u_label_sem[:FLAGS.limit]

        # open prediction
        pred = SemLaserScan(sem_color_dict=LAB_PARAMS['color_map'],
                            project=False,
                            H=sensor_img_H,
                            W=sensor_img_W,
                            fov_up=sensor_fov_up,
                            fov_down=sensor_fov_down,
                            max_classes=300)
        pred.open_scan(scan_file_path)
        pred.open_label(pred_file_path)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_pred_sem = u_pred_sem[:FLAGS.limit]

        # add single scan to evaluation
        evaluator.addBatch(u_pred_sem, u_label_sem)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    print('{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=split,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))

    save_to_log(FLAGS.predictions, pred_log_file, '{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(split=split,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
            save_to_log(FLAGS.predictions, pred_log_file, 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

    # print for spreadsheet
    print('*' * 80)
    print('below can be copied straight for paper table')
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(',')
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(',')
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    # possible splits
    poss_splits = ['train', 'valid', 'test']

    parser = argparse.ArgumentParser('./evaluate_iou.py')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--data_cfg', '-f',
        type=str,
        required=True,
        help='Data augmentations and label configurations. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=None,
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=['train', 'valid', 'test'],
        default=None,
        help='Split to evaluate on. One of ' +
             str(poss_splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print('*' * 80)
    print('INTERFACE:')
    print('Data: ', FLAGS.dataset)
    print('data_cfg', FLAGS.data_cfg)
    print('Predictions: ', FLAGS.predictions)
    print('Split: ', FLAGS.split)
    print('Limit: ', FLAGS.limit)
    print('*' * 80)

    # assert split
    assert(FLAGS.split in poss_splits)

    # open navigation file
    nav_file_path = path.join(FLAGS.data_cfg, 'navigation.yml')
    print('Opening navigation file %s' % nav_file_path)
    NAV = yaml.safe_load(open(nav_file_path, 'r'))

    # open dataset stats file
    try:
        sensors_file_path = path.join(FLAGS.data_cfg, 'dataset_params.yml')
        print('Opening dataset stats file %s' % sensors_file_path)
        SENSORS = yaml.safe_load(open(sensors_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening dataset stats file.')
        quit()

    # open data stats file
    stats_file_path = path.join(FLAGS.data_cfg, 'data_stats.yml')
    print('Opening data stats file %s' % stats_file_path)
    STATS = yaml.safe_load(open(stats_file_path, 'r'))

    # open label params file
    lab_params_file_path = path.join(FLAGS.data_cfg, 'label_params.yml')
    print('Opening label params file %s' % lab_params_file_path)
    LAB_PARAMS = yaml.safe_load(open(lab_params_file_path, 'r'))

    # get number of interest classes, and the label mappings
    class_strings = LAB_PARAMS['labels']
    class_remap = LAB_PARAMS['learning_map']
    class_inv_remap = LAB_PARAMS['learning_map_inv']
    class_ignore = LAB_PARAMS['learning_ignore']
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = 0
    for key, data in class_remap.items():
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in class_remap.items():
        try:
            remap_lut[key] = data
        except IndexError:
            print('Wrong key ', key)

    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print('Ignoring xentropy class ', x_cl, ' in IoU evaluation')

    # create evaluator
    device = torch.device('cpu')
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()

    # get test set
    if FLAGS.split is not None:
        evaluate(FLAGS.dataset, NAV, STATS, SENSORS, LAB_PARAMS, FLAGS.split, FLAGS.predictions)
    else:
        for split in ('train','valid'):
            evaluate(FLAGS.dataset, NAV, STATS, SENSORS, LAB_PARAMS, split, FLAGS.predictions)

# ./evaluate_iou.py -d /Dataset/ -f ./../../../cfgs -p ./../../../logs/infer/ddrnet_aug5_T/ --split valid
