import caffe
import numpy as np
import matplotlib.pyplot as plt
#from misc.utils import *
from functions import *
from caffe_VGG_19 import *
from scipy.optimize import minimize
from skimage.io import imsave

caffe.set_device(0)
caffe.set_mode_gpu()

model_path = "model/VGG_ILSVRC_19_layers_deploy.prototxt"
train_path = "model/VGG_ILSVRC_19_layers.caffemodel"
mean_path = "models/ilsvrc_2012_mean.npy"

net = caffe.Net(model_path, train_path, caffe.TEST)
VGG_MEAN = zip('BGR', np.load(mean_path))

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', VGG_MEAN)
transformer.set_raw scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

photo = caffe.io.load_image("../Images/dieux_du_stade.jpg")
plt.imshow(photo)

transformed_photo = transformer.preprocess('data', photo)
net.blobs['data'].data[...] = transformed_photo
