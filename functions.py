import caffe
import numpy as np
from scipy.linalg.blas import sgemm
import copy

def structure_loss(F, P):
    Fc = F[0]
    Pc = P[0]
    channel, row, col = Fc.shape
    Fc = Fc.reshape((channel, row*col))
    Pc = Pc.reshape((channel, row*col))

    loss = 0.5 * np.sum((Fc-Pc)**2)
    grad = (Fc-Pc) * (Fc > 0)

    return loss, grad


def style_loss(F, A, layer, style_layers):
    idx = style_layers.index(layer)+1

    Fl = np.squeeze(F[idx])
    Al = np.squeeze(A[idx])

    channel, row, col = Fl.shape
    Fl = Fl.reshape((channel, row*col))
    Al = Al.reshape((channel, row*col))

    gram_F = sgemm(1, Fl, Fl.T)
    gram_A = sgemm(1, Al, Al.T)

    denom = (2*channel*row*col)**2
    loss = np.sum((gram_F-gram_A)**2) / denom
    grad = 4 * sgemm(1, gram_F-gram_A, Fl) * (Fl > 0) / denom

    return loss, grad


def fun_opt(x, A, P, net, style_layers, content_layer, start_layer, \
            alpha, beta, step):
    x = x.reshape(net.blobs["data"].data.shape[1:])
    net.blobs["data"].data[0,:,:,:] = x

    net.forward()

    F = [net.blobs[content_layer].data[0].copy()]
    for layer in style_layers:
        F.append(net.blobs[layer].data[0].copy())

    layers = copy.copy(style_layers)
    if layers[-1] == "conv5_1":
        layers.insert(len(layers)-1, content_layer)
    else:
        layers.append(content_layer)
    idx = len(layers)-1

    w = {"conv1_1": 0.2, "conv2_1": 0.2, "conv3_1": 0.2, "conv4_1":0.2, "conv5_1":0.2}

    loss = 0
    net.blobs[layers[-1]].diff[:] = 0

    for layer in layers[::-1]:
        grad = net.blobs[layer].diff[0]
        if idx > 0:
            next_layer = layers[idx-1]
        else:
            next_layer = None

        if layer in style_layers:
            loss_s, grad_s = style_loss(F, A, layer, style_layers)
            loss += beta*w[layer]*loss_s
            net.blobs[layer].diff[0] += beta*0.2*grad_s.reshape(grad.shape)

        if layer == content_layer:
            loss_c, grad_c = structure_loss(F, P)
            loss += alpha*loss_c
            net.blobs[layer].diff[0] += alpha*grad_c.reshape(grad.shape)

        out = net.backward(start = layer, end = next_layer)

        if next_layer is None:
            grad = net.blobs["data"].diff[0]
            break
        else:
            grad = net.blobs[next_layer].diff[0]

        idx -= 1

    return loss, grad.flatten().astype(np.float64)


def get_transformation(img, net, mean):
    row, col, channel = img.shape

    transformer = caffe.io.Transformer({'data': net.blobs["data"].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale("data", 255)

    img_out = transformer.preprocess('data', img)

    return img_out, transformer
