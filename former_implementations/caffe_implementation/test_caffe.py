import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from scipy.optimize import minimize

caffe.set_device(0)
caffe.set_mode_gpu()

#net = caffe.Net('conv.prototxt', caffe.TEST)
model_path = "../model/VGG_ILSVRC_19_layers_deploy.prototxt"
train_path = "../model/VGG_ILSVRC_19_layers.caffemodel"
mean_path = "../model/ilsvrc_2012_mean.npy"

net = caffe.Net(model_path, train_path, caffe.TEST)
mean=np.load(mean_path).mean(1).mean(1)

x = caffe.io.load_image("../Images/marie.jpg")

transformer = caffe.io.Transformer({'data': net.blobs["data"].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', mean)
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale("data", 255)

img = transformer.preprocess('data', x)

net.blobs["data"].data[...] = img

net.forward()

output = net.blobs["fc7"].data[0]

print(np.amax(net.blobs["conv4_2"].diff[0]))

net.backward()

print(np.amax(net.blobs["conv4_2"].diff[0]))
