#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom-VGG import Vgg19
import utils

photo = utils.load_image("./Images/argetine.jpg")
art   = utils.load_image("./Images/nuit_etoilee.jpg")

if __name__ = '__main__':
    npy_path = 'tensorflow-vgg/'
    CNN = Vgg19(npy_path)
