#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import random
import numpy as np
import torch

from torch.utils.data import Dataset



class UTDdataset(Dataset):
    def __init__(self,**kwargs):
        super(UTDdataset, self).__init__()
        self.labels=kwargs['labels']
        self.depth=kwargs['depth']
        self.inertial=kwargs['inertial']
        self.skeleton=kwargs['skeleton']
        self.color=kwargs['color']
        self.__normalize__()


    def __normalize__(self):
        self.depth[self.depth != self.depth] = 0
        self.inertial[self.inertial != self.inertial] = 0
        self.skeleton[self.skeleton != self.skeleton] = 0
        self.color[self.color!=self.color]=0
        self.depth=self.depth/np.max(self.depth)
        self.inertial=self.inertial/np.max(self.inertial)
        self.skeleton=self.skeleton/np.max(self.skeleton)
        self.color=self.color/np.max(self.color)

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

