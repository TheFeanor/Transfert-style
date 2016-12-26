import caffe
import numpy as np
import matplotlib.pyplot as plt
#from misc.utils import *
from functions import *
from caffe_VGG_19 import *
from skimage.io import imsave

N = 5 # number of iterations of the GD
layer = "conv4_2" # index of the layer

photo_path = "../Images/dieux_du_stade.jpg"

caffe.set_device(0)
caffe.set_mode_gpu()

model_path = "model/VGG_ILSVRC_19_layers_deploy.prototxt"
train_path = "model/VGG_ILSVRC_19_layers.caffemodel"
mean_path = "model/ilsvrc_2012_mean.npy"

net = caffe.Net(model_path, train_path, caffe.TEST)
mean=np.load(mean_path).mean(1).mean(1)/255

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mean)
transformer.set_channel_swap('data', (2, 1, 0))

photo = caffe.io.load_image(photo_path)
plt.imshow(photo)

transformed_input = transformer.preprocess('data', photo)
net.blobs['data'].data[...] = transformed_input

net.forward()
p_features = net.blobs[layer].data[0, :]
# print(p_features.shape)

row, col, channel = photo.shape
output = np.random.rand(row, col, channel)
transformed_output = transformer.preprocess('data', output)

net.blobs['data'].data[...] = transformed_output
net.forward()

o_features = net.blobs[layer].data[0, :]
# print(p_features.shape)

min_args = {
    "args": (o_features, p_features, net, layer), "method": "L-BFGS-B",
    "jac": True, "options":{"maxiter": 5}
}
res = minimize(compute_inverse_loss, transformed_output, **min_args)
transformed_output = net.blobs["data"].data
output = transformer.deprocess('data', transformed_output)

imsave("../Images/inverse/output.jpg", output)
