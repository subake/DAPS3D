#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import yaml
import os
from os import path

from tasks.semantic.modules.user import *


if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--data_cfg', '-f',
        type=str,
        required=True,
        help='Data augmentations and label configurations. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
                datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--model_name', '-n',
        type=str,
        required=True,
        default='salsanext'
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print('data_cfg', FLAGS.data_cfg)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("infering", FLAGS.split)
    print("----------\n")

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # open arch config file
    print("Opening arch config file from %s" % FLAGS.model)
    ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))

    # open navigation file
    nav_file_path = path.join(FLAGS.data_cfg, 'navigation.yml')
    print('Opening navigation file %s' % nav_file_path)
    NAV = yaml.safe_load(open(nav_file_path, 'r'))

    # open data stats file
    try:
        stats_file_path = path.join(FLAGS.model, 'data_stats.yaml')
        print('Opening data stats file %s' % stats_file_path)
        STATS = yaml.safe_load(open(stats_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening data stats file.')
        quit()

    # open dataset stats file
    try:
        sensors_file_path = path.join(FLAGS.data_cfg, 'dataset_params.yml')
        print('Opening dataset stats file %s' % sensors_file_path)
        SENSORS = yaml.safe_load(open(sensors_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening dataset stats file.')
        quit()

    # open augment file
    try:
        augment_file_path = path.join(FLAGS.model, 'augmentation.yaml')
        print('Opening data augmentation file %s' % stats_file_path)
        AUGMENT = yaml.safe_load(open(augment_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening data stats file.')
        quit()

    # open label params file
    lab_params_file_path = path.join(FLAGS.model, 'label_params.yaml')
    print('Opening label params file %s' % lab_params_file_path)
    LAB_PARAMS = yaml.safe_load(open(lab_params_file_path, 'r'))

    if not os.path.isdir(FLAGS.log):
        print('Error: output directory does not exist!')
        print(FLAGS.log)
        quit()

    # create user and infer dataset
    user = User(ARCH, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, FLAGS.dataset,
                FLAGS.log, FLAGS.model, FLAGS.model_name, FLAGS.split)
    user.infer()
