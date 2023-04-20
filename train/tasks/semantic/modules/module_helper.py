#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            if torch_ver in ('1.0', '1.1'):
                from tasks.semantic.modules.inplace_abn.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
                