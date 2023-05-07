#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
from os import path
import shutil
from shutil import copyfile
import yaml
import __init__ as booger
from tasks.semantic.modules.trainer import *
from pip._vendor.distlib.compat import raw_input

if __name__ == '__main__':
    parser = argparse.ArgumentParser('./train.py')
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
        '--arch_cfg', '-ac',
        type=str,
        required=True,
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default='~/output',
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default=None,
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Which model to train'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # setting log path
    FLAGS.log = path.join(FLAGS.log, 'logs', datetime.datetime.now().strftime('%Y-%-m-%d-%H:%M'))

    # checking model
    if FLAGS.model not in ('salsanet','salsanext', 'salsanet_rec', 'salsanet_rec_lstm', 'ddrnet', 'segformer'):
        print('Flags model:',FLAGS.model)
        raise NotImplementedError('you need to chose between: salsanet, salsanet_rec, salsanet_rec_lstm, salsanext, ddrnet or segformer')
    
    # print summary of what we will do
    print('----------')
    print('INTERFACE:')
    print('dataset', FLAGS.dataset)
    print('arch_cfg', FLAGS.arch_cfg)
    print('data_cfg', FLAGS.data_cfg)
    print('log', FLAGS.log)
    print('Model', FLAGS.model)
    print('pretrained', FLAGS.pretrained)
    print('----------\n')

    # open arch config file
    try:
        print('Opening arch config file %s' % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print('Error opening arch yaml file.')
        quit()

    # open navigation file
    try:
        nav_file_path = path.join(FLAGS.data_cfg, 'navigation.yml')
        print('Opening navigation file %s' % nav_file_path)
        NAV = yaml.safe_load(open(nav_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening navigation file.')
        quit()

    # open data stats file
    try:
        stats_file_path = path.join(FLAGS.data_cfg, 'data_stats.yml')
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
        augment_file_path = path.join(FLAGS.data_cfg, 'augmentation.yml')
        print('Opening data augmentation file %s' % stats_file_path)
        AUGMENT = yaml.safe_load(open(augment_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening data stats file.')
        quit()

    # open label params file
    try:
        lab_params_file_path = path.join(FLAGS.data_cfg, 'label_params.yml')
        print('Opening label params file %s' % lab_params_file_path)
        LAB_PARAMS = yaml.safe_load(open(lab_params_file_path, 'r'))
    except Exception as e:
        print(e)
        print('Error opening label params file.')
        quit()

    # create log folder
    try:
        if FLAGS.pretrained is '':
            FLAGS.pretrained = None
            if os.path.isdir(FLAGS.log):
                if os.listdir(FLAGS.log):
                    answer = raw_input('Log Directory is not empty. Do you want to proceed? [y/n]  ')
                    if answer == 'n':
                        quit()
                    else:
                        shutil.rmtree(FLAGS.log)
            os.makedirs(FLAGS.log)
        else:
            FLAGS.log = FLAGS.pretrained
            print('Not creating new log file. Using pretrained directory')
    except Exception as e:
        print(e)
        print('Error creating log directory. Check permissions!')
        quit()

    # does model folder exist?
    if FLAGS.pretrained is not None:
        if os.path.isdir(FLAGS.pretrained):
            print('model folder exists! Using model from %s' % (FLAGS.pretrained))
        else:
            print('model folder doesnt exist! Start with random weights...')
    else:
        print('No pretrained directory found.')

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print('Copying files to %s for further reference.' % FLAGS.log)
        copyfile(FLAGS.arch_cfg, FLAGS.log + '/arch_cfg.yaml')
        copyfile(stats_file_path, FLAGS.log + '/data_stats.yaml')
        copyfile(sensors_file_path, FLAGS.log + '/dataset_params.yaml')
        copyfile(nav_file_path, FLAGS.log + '/navigation.yaml')
        copyfile(lab_params_file_path, FLAGS.log + '/label_params.yaml')
        copyfile(augment_file_path, FLAGS.log + '/augmentation.yaml')
    except Exception as e:
        print(e)
        print('Error copying files, check permissions. Exiting...')
        quit()

    # create trainer and start the training
    trainer = Trainer(ARCH, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, FLAGS.dataset, FLAGS.log, FLAGS.pretrained, FLAGS.model)
    trainer.train()
