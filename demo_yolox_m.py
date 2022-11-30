#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/content/demo_obj_detection"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_test2017.json"
        self.input_size = (640,640)
        self.num_classes = 1
        self.test_size = (640,640) 
        self.max_epoch = 100
