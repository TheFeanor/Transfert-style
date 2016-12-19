import caffe
import numpy as np
import matplotlib.pyplot as plt
#from misc.utils import *
from functions import *
from caffe_VGG_19 import *
from skimage.io import imsave

N = 50 # number of iterations of the GD
layer = "conv4_2" # index of the layer

photo_path = "../Images/dieux_du_stade.jpg"

caffe.set_device(0)
caffe.set_mode_gpu()

model_path = "model/VGG_ILSVRC_19_layers_deploy.prototxt"
train_path = "model/VGG_ILSVRC_19_layers.caffemodel"
mean_path = "model/ilsvrc_2012_mean.npy"

net = caffe.Net(model_path, train_path, caffe.TEST)
mean=np.load(mean_path).mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

photo = caffe.io.load_image(photo_path)
plt.imshow(photo)

transformed_inputs = transformer.preprocess('data', photo)
net.blobs['data'].data[...] = transformed_inputs

net.forward()

p_features = net.blobs[layer].data[0, :]

row, col, channel = photo.shape
output = np.random.rand(row, col, channel)
output_transformed = transformer.preprocess('data', output)

for step in np.arange(N):
    net.blobs['data'].data[...] = output_transformed
    net.forward()

    o_features = net.blobs[layer].data[:]

    loss, grad = compute_inverse_loss(o_features, p_features, \
                                      output_transformed, net, I)

    output_transformed = minimize(loss, output_transformed, grad)

output = transformer.deprocess('data', output_transformed)

imsave("../Images/inverse/output.jpg", output)
