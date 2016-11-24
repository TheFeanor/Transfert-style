#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow-vgg.vgg19 as vgg
import matplotlib as plt
import cv

if __name__ = '__main__':
    npy_path = 'tensorflow-vgg/'
    vgg19 = vgg.Vgg19(npy_path)
