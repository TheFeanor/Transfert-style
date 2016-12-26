import os
import sys
import args

import caffe
import numpy as np



#caffe.set_device(0)
#caffe.set_mode_gpu()

class my_net(object):

    def __init__(self):

        mean_path = "model/ilsvrc_2012_mean.npy"

        self.net = load_model(model_path, train_path, mean_path)

        print(self.net.blobs)
