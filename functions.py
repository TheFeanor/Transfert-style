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
    #print("diff at input : {}".format(np.linalg.norm(net.blobs["data"].data[0] - x)))

    net.forward()

    F = [net.blobs[content_layer].data[0].copy()]
    for layer in style_layers:
        F.append(net.blobs[layer].data[0].copy())
    print(map(lambda a: np.mean(a), F))
    layers = copy.copy(style_layers)
    if layers[-1] == "conv5_1":
        layers.insert(len(layers)-1, content_layer)
    else:
        layers.append(content_layer)
    idx = len(layers)-1

    loss = 0
    net.blobs[layers[-1]].diff[:] = 0

    for layer in layers[::-1]:
        grad = net.blobs[layer].diff[0]
        if idx > 0:
            next_layer = layers[idx-1]
        else:
            next_layer = None
        #print(layer)
        #print("norm of top : {}".format(np.linalg.norm(net.blobs[layer].diff[2])))

        if layer in style_layers:
            loss_s, grad_s = style_loss(F, A, layer, style_layers)
            loss += beta*0.2*loss_s
            net.blobs[layer].diff[0] += beta*0.2*grad_s.reshape(grad.shape)

        if layer == content_layer:
            loss_c, grad_c = structure_loss(F, P)
            loss += alpha*loss_c
            net.blobs[layer].diff[0] += alpha*grad_c.reshape(grad.shape)

        #print("norm of top after layer {} update: {}".format(layer, np.linalg.norm(net.blobs[layer].diff[2])))

        out = net.backward(start = layer, end = next_layer)
        # print("output of backward : {}".format(out.keys()))
        # print("output shape : {}".format(out.values()[0].shape))
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
            break
        else:
            grad = net.blobs[next_layer].diff[0]

        #print("norm of bottom : {}".format(np.linalg.norm(grad)))
        idx -= 1
    print("norm of data : {}".format(np.log10(np.linalg.norm(grad))))
    return loss, grad.flatten().astype(np.float64)


    def gradient_descent(loss, grad, step = 1e-2):
        return loss - step*grad
