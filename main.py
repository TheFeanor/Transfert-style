#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom-VGG import Vgg19
import utils

photo = utils.load_image("./Images/argetine.jpg")
art   = utils.load_image("./Images/nuit_etoilee.jpg")

photo_batch = photo.reshape((1,224,224,3))
art_batch   = art.reshape((1,224,224,3)) 

with tf.device('/cpu:0'):
    npy_path = 'tensorflow-vgg/'
    CNN = Vgg19(npy_path)
