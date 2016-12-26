import caffe
import numpy as np
from scipy.linalg.blas import sgemm
from scipy.optimize import minimize
from skimage.io import imsave
import functions as fc

caffe.set_device(0)
caffe.set_mode_gpu()

alpha = 1
beta = 5e3
N = 500

style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
content_layer = "conv4_2"
start_layer = style_layers[-1]
#start_layer = content_layer

model_path = "model/VGG_ILSVRC_19_layers_deploy.prototxt"
train_path = "model/VGG_ILSVRC_19_layers.caffemodel"
mean_path = "model/ilsvrc_2012_mean.npy"

net = caffe.Net(model_path, train_path, caffe.TEST)
mean=np.load(mean_path).mean(1).mean(1)

photo_path = "Images/marie.jpg"
art_path = "Images/filles_piano.jpg"

photo = caffe.io.load_image(photo_path)
art = caffe.io.load_image(art_path)

net.blobs["data"].reshape(1, 3, 224, 224)

transformed_input, transformer = fc.get_transformation(photo, net, mean)
net.blobs['data'].data[0] = transformed_input
net.forward()
P = [net.blobs[content_layer].data[0].copy()]
for layer in style_layers:
    P.append(net.blobs[layer].data[0].copy())

transformed_input2, _ = fc.get_transformation(art, net, mean)

net.blobs['data'].data[0, :, :, :] = transformed_input2
net.forward()
A = [net.blobs[content_layer].data[0].copy()]
for layer in style_layers:
    A.append(net.blobs[layer].data[0].copy())

#print np.amax(transformed_input - transformed_input2)
print(np.amax(A[2]-P[2]))
print(map(lambda x: np.mean(x), P))
print(map(lambda x: np.mean(x), A))


row, col, channel = photo.shape
# bnds = np.array([(0,255) for i in np.arange(row*col*channel)], dtype=(np.int32, np.int32))
# bnds = bnds.reshape((row, col, channel))

x = 0.95*transformed_input + 0.05*transformed_input2

# compute data bounds
data_min = -transformer.mean["data"][:,0,0]
data_max = data_min + transformer.raw_scale["data"]
bnds = [(data_min[0], data_max[0])]*(x.size/3) + \
              [(data_min[1], data_max[1])]*(x.size/3) + \
              [(data_min[2], data_max[2])]*(x.size/3)

step = 0

def callback(xk):
    global step
    step += 1

    if step % 20 == 0:
        print("Step {}".format(step))

min_args = {
    "args": (A, P, net, style_layers, content_layer, start_layer, alpha, beta, step),
    "method": "L-BFGS-B", "jac" : True, "bounds": bnds,
    "options": {"maxiter": N}, "callback": callback
}

res = minimize(fc.fun_opt, x, **min_args)

# for step in np.arange(2):
#     if step % 10 == 0:
#         print("step : {}".format(step))
#
#     loss, grad = fc.fun_opt(x, A, P, net, style_layers, content_layer, \
#                             start_layer, alpha, beta)
#     x = x - 0.00002 * grad

#out = net.blobs["data"].data[0, :, :, :]
out = net.blobs["data"].data[0]
img_out = transformer.deprocess('data', out)
#print("max out : {}".format(np.amax(out)))

imsave("Images/transfer/marie_filles.jpg", img_out)
