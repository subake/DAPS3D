#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from tqdm import tqdm

from tasks.semantic.modules.SalsaNextRecLSTM import SalsaNextRecLSTM
from tasks.semantic.modules.SalsaNetRecLSTM import SalsaNetRecLSTM
from tasks.semantic.modules.SalsaNetRec import SalsaNetRec
from tasks.semantic.modules.SalsaNext import SalsaNext
from tasks.semantic.modules.segmentator import SalsaNet

from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.dataset.kitti.parser import Parser

from tasks.semantic.modules.criterion import OhemCrossEntropy


class User():
    def __init__(self, ARCH, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, datadir, logdir, modeldir, modelname, split):
        # parameters
        self.ARCH = ARCH
        self.NAV = NAV
        self.LAB_PARAMS = LAB_PARAMS
        self.AUGMENT = AUGMENT
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.modelname = modelname
        self.split = split

        if self.modelname == 'ddrnet':
            if ARCH['MODEL']['MOD'] == 'oc':
                from tasks.semantic.modules.ddrnet_23_slim_salsa_oc  import get_seg_model, FullModel_infer
            elif ARCH['MODEL']['MOD'] == 'da':
                from tasks.semantic.modules.ddrnet_23_slim_salsa_da  import get_seg_model, FullModel_infer
            else:
                from tasks.semantic.modules.ddrnet_23_slim_salsa  import get_seg_model, FullModel_infer
        elif self.modelname == 'segformer':
            from tasks.semantic.modules.segformer_salsa import get_seg_model, FullModel_infer

        # get the data
        self.parser = Parser(self.datadir, ARCH, NAV, STATS, SENSORS, LAB_PARAMS, AUGMENT, batch_size=1)

        # concatenate the encoder and the head
        if 'salsa' in self.modelname:
            with torch.no_grad():
                print('modeldir: %s' % self.modeldir)
                model_path = os.path.join(self.modeldir, 'SalsaNet_valid_best')
                print('model_path: %s' % model_path)

                if self.modelname == 'salsanet':
                    self.model = SalsaNet(self.parser.get_n_classes())
                elif self.modelname == 'salsanext':
                    self.model = SalsaNext(self.parser.get_n_classes())
                elif self.modelname == 'salsanet_rec':
                    self.model = SalsaNetRec(self.parser.get_n_classes())
                elif self.modelname == 'salsanet_rec_lstm':
                    self.model = SalsaNetRecLSTM(self.parser.get_n_classes())
                elif self.modelname == 'salsanext_rec_lstm':
                    self.model = SalsaNextRecLSTM(self.parser.get_n_classes())
                
                self.model = nn.DataParallel(self.model)

                torch.nn.Module.dump_patches = True

                w_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

                self.model.cpu()

                # NEED TO SWITCH if no DataParallel
                self.model.module.load_state_dict(w_dict['state_dict'], strict=True)
                
        elif self.modelname == 'segformer':
            with torch.no_grad():
                print('modeldir: %s' % self.modeldir)
                model_path = os.path.join(self.modeldir, 'SegFormer_valid_best')
                print('model_path: %s' % model_path)

                self.model = get_seg_model(self.ARCH["MODEL"]["BACKBONE"], self.parser.get_n_classes())

                criterion = OhemCrossEntropy(ignore_label=0,
                                        thres=self.ARCH["LOSS"]["OHEMTHRES"],
                                        min_kept=self.ARCH["LOSS"]["OHEMKEEP"],
                                        weight=self.ARCH["DATASET"]["CLASS_WEIGHTS"],
                                        config=self.ARCH)
                self.model = FullModel_infer(self.model, criterion)
                self.model = nn.DataParallel(self.model)

                torch.nn.Module.dump_patches = True

                w_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

                self.model.cpu()

                state_dict = w_dict['state_dict']

                self.model.module.load_state_dict(state_dict, strict=True)

        elif self.modelname == 'ddrnet':
            with torch.no_grad():
                print('modeldir: %s' % self.modeldir)
                model_path = os.path.join(self.modeldir, 'DDRNet_valid_best')
                print('model_path: %s' % model_path)

                self.model = get_seg_model(self.parser.get_n_classes(), channels=self.ARCH["MODEL"]["CHANNELS"], mode='train')

                criterion = OhemCrossEntropy(ignore_label=0,
                                        thres=self.ARCH["LOSS"]["OHEMTHRES"],
                                        min_kept=self.ARCH["LOSS"]["OHEMKEEP"],
                                        weight=self.ARCH["DATASET"]["CLASS_WEIGHTS"],
                                        config=self.ARCH)
                self.model = FullModel_infer(self.model, criterion)
                self.model = nn.DataParallel(self.model)

                torch.nn.Module.dump_patches = True

                w_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

                self.model.cpu()

                state_dict = w_dict['state_dict']
                
                self.model.module.load_state_dict(w_dict['state_dict'], strict=True)
                
        else:
            with torch.no_grad():
                self.model = SalsaNet(self.ARCH,
                                         self.parser.get_n_classes(),
                                         self.modeldir)

        # use knn post processing?
        self.post = None
        if self.ARCH['post']['KNN']['use']:
            self.post = KNN(self.ARCH['post']['KNN']['params'], self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Infering in device: ', self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()


    def infer(self):
        if self.split == None:
            # do train set
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original)
            # do valid set
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original)
            # do test set
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original)
        elif self.split == 'valid':
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original)
        elif self.split == 'train':
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original)
        else:
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original)

        print('Finished Infering')

        return

    def infer_subset(self, loader, to_orig_fn):
        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            times = []
            infer_starter = torch.cuda.Event(enable_timing=True)
            infer_ender = torch.cuda.Event(enable_timing=True)
            total_starter = torch.cuda.Event(enable_timing=True)
            total_ender = torch.cuda.Event(enable_timing=True)
            # TOTAL TIME START
            total_starter.record()
            for i, (proj_in, proj_mask, _, _, path_seq, path_name, dataset_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints, proj_time) in enumerate(tqdm(loader)):
                # first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                # loading data on GPU
                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                # INFER TIME START
                infer_starter.record()

                # computing output
                proj_output = self.model(proj_in)

                # INFER TIME END
                infer_ender.record()
                torch.cuda.synchronize()
                infer_time = infer_starter.elapsed_time(infer_ender)

                # getting raw classes
                proj_argmax = proj_output[0].argmax(dim=0)

                # postprocessing
                if self.post:
                    # knn postproc
                    unproj_argmax = self.post(proj_range, unproj_range, proj_argmax, p_x, p_y)
                else:
                    # put in original pointcloud using indexes
                    unproj_argmax = proj_argmax[p_y, p_x]

                # get the first scan in batch and projecting
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # mapping to original label
                pred_np = to_orig_fn(pred_np)

                # TOTAL TIME END
                total_ender.record()
                torch.cuda.synchronize()
                total_time = total_starter.elapsed_time(total_ender)

                # save label
                save_path = os.path.join(self.logdir, dataset_name[0], path_seq)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, path_name)
                np.save(save_path, pred_np)

                # saving time
                times.append([total_time, infer_time])
                # TOTAL TIME START
                total_starter.record()

            print('*' * 30)
            times = np.array(times)[1:]
            for ind, stat_name in enumerate(['Total', 'Infer']):
                print('-' * 10)
                print('%s time:' % stat_name)
                print('\tMean (ms): %s' % np.mean(times[:, ind]))
                print('\tStd: %s' % np.std(times[:, ind]))

    def predict(self):
        pass
     

