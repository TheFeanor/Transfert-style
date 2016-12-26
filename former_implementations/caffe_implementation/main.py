import caffe
import numpy as np
import matplotlib.pyplot as plt
#from misc.utils import *
from functions import *
from caffe_VGG_19 import *
from skimage.io import imsave

layer = "conv3_1" # index of the layer
I = 3
alpha = 1
beta = 1e4

photo_path = "../Images/dieux_du_stade.jpg"
art_path = "../Images/nuit_etoilee.jpg"

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
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale("data", 255)

photo = caffe.io.load_image(photo_path)
art = caffe.io.load_image(art_path)
plt.imshow(photo)
plt.imshow(art)

feat_maps = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

transformed_input = transformer.preprocess('data', photo)
net.blobs['data'].data[...] = transformed_input
net.forward()
p_features = []
for layer in feat_maps:
    p_features.append(net.blobs[layer].data[0, :])

transformed_input = transformer.preprocess('data', art)
net.blobs['data'].data[...] = transformed_input
net.forward()
a_features = []
for layer in feat_maps:
    a_features.append(net.blobs[layer].data[0, :])

img = transformer.preprocess('data', photo)

# compute data bounds
data_min = -transformer.mean["data"][:,0,0]
data_max = data_min + transformer.raw_scale["data"]
data_bounds = [(data_min[0], data_max[0])]*(img.size/3) + \
              [(data_min[1], data_max[1])]*(img.size/3) + \
              [(data_min[2], data_max[2])]*(img.size/3)

min_args = {
    "args": (p_features, a_features, net, layer, I, alpha, beta), "method": "L-BFGS-B",
    "jac": True, "options":{"maxiter": 512}
}

res = minimize(compute_transfer_loss, img.flatten(), **min_args)

output_transformed = net.blobs["data"].data
output = transformer.deprocess('data', output_transformed)

imsave("../Images/out/output.jpg", output)
