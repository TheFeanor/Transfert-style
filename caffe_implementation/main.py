import caffe
import numpy as np
from misc.utils import *
from functions import *
from scipy.optimize import minimize
from skimage.io import imsave

caffe.set_device(0)
caffe.set_mode_gpu()
