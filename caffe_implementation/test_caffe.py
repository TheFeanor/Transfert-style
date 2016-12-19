import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('conv.prototxt', caffe.TEST)
# model_path = "model/VGG_ILSVRC_19_layers_deploy.prototxt"
# train_path = "model/VGG_ILSVRC_19_layers.caffemodel"
# mean_path = "models/ilsvrc_2012_mean.npy"
#
# net = caffe.Net(model_path, train_path, caffe.TEST)

# for k, v in net.blobs.items():
#     print("({}, {})".format(k, v.data.shape))
#
# for k, v in net.params.items():
#     print("({}, {}, {})".format(k, v[0].data.shape, v[1].data.shape))
#
# print net.blobs['conv'].data.shape

#print type(net.blobs['conv'].diff)
#print net.blobs['conv'].diff.shape
#print net.blobs['conv'].diff[0]
print net.blobs.keys()

layer = net.blobs["conv"]
print layer
