import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

def features_distance(A, B):
    """
    Computes the Euclidian distance between two features maps A and B.
    The parameters is_square_norm denotes wether the output is the distance
    or the square distance.
    """
    distance = norm(A-B)**2

    return distance


def style_representation(A):
    """
    Computes the style representation of the feature map A along axis 3 that
    represents the amount of filter responses.
    """
    A = np.squeeze(A)
    channel, row, col = A.shape
    Gram = np.matrix(np.zeros((row, col)))

    for i in np.arange(channel):
        for j in np.arange(i, channel, 1):
            # computes correlation between response of filter i and j
            product = A[i, :, :] * A[j, :, :]
            Gram[i,j] = np.sum(product)

    Gram = Gram + Gram.T - np.diag(Gram.diagonal())

    return Gram


def style_error(features_a, features_x, w = np.ones(5)/5):
    """
    Computes style error based on the definition of the paper.
    w is a vector of weights of size (1,5)
    """
    E = np.zeros(5)
    grad = np.zeros(5)
    channel, row, col = features_a.shape

    for k in np.arange(5):
        N_I = channel
        M_I = row * col

        A = style_representation(features_a[k])
        G = style_representation(features_x[k])

        F = np.matrix(features_x[k])
        F = np.reshape(F, (M_I, N_I))

        diff = A - G
        E[k] = w[k] / (4* M_I**2 * N_I**2) * norm(np.inner(F, diff)) ** 2
        grad[k] = w[k] / (M_I**2 * N_I**2) * diff * (G > 0)

    return E, grad


def structure_error(features_p, features_x, mode):
    """
    Computes structure_error based on the definiton of the paper.
    """
    E = 0
    print(features_p.shape)
    channel, row, col = features_p.shape

    if mode == "transfer":
        P = features_p[-1]
        F = features_x[-1]
    else:
        P = features_p
        F = features_x

    N_I = channel
    M_I = row * col

    print(P.shape)

    P = np.reshape(P, [M_I, N_I])
    F = np.reshape(F, [M_I, N_I])

    grad = P-F
    E = 0.5 * norm(grad) ** 2
    grad = grad * (F > 0)

    return E, grad


def alpha_reg(x, alpha, lambd):
    """
    Computes alpha-norm-regularisation term with weight lambd
    """
    row, col, channel  = x.shape
    norm_a = 0
    grad_a = np.zeros(x.shape)
    eps = 1e-6

    for k in np.arange(channel):
        xt = np.reshape(x[:, :, k], row*col)
        x_c = xt - np.mean(xt)
        norm_a += np.sum(np.power(x_c, alpha))

        for i in np.arange(row):
            for j in np.arange(col):
                num = (x[i, j, k]+eps)**alpha - x[i, j, k]**alpha
                grad_a[i, j, k] = num / eps

    norm_a *= lambd
    grad_a *= lambd

    return norm_a, grad_a


def TV_reg(x, beta, lambd):
    """
    Computes TV-regularisation term with power beta/2 and weight lambd
    """
    row, col, channel  = x.shape
    norm_b = 0
    grad_b = np.zeros(x.shape)

    for k in np.arange(channel):
        xt = x[:, :, k]
        grad_uv = np.gradient(xt)

        beta2 = np.float(beta)/2
        norm_temp = np.sum(np.power(grad_uv[0], 2) + np.power(grad_uv[1], 2))
        norm_b += np.power(norm_temp, beta2)

        for i in np.arange(row):
            for j in np.arange(col):
                grad_b[i, j, k] = beta*x[i, j, k]*np.power(norm(xt), beta/2-1)

    norm_b *= lambd
    grad_b += lambd

    return norm_b, grad_b


def getBottomLayers(idx, net):
    all_layers = net.blobs.keys()
    layers = all_layers[:idx+1][::-1]

    indexes = np.arange(len(layers))[::-1]

    return layers, indexes


def compute_transfer_loss(o_features, p_features, a_features, net, I, alpha, beta):
    style_loss, style_grad = style_error(a_features, o_features, mode = "transfer")
    structure_loss, structure_grad = structure_error(p_features, o_features)

    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    structure_layer = "conv4_2"
    layers, indexes = getBottomLayers(I, net)
    compt_style_layer = int(I[4])-1

    loss = 0
    net.blobs[layers[-1]].diff[:] = 0

    for i, layer in enumerate(layers):
        next_layer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]

        if layer in style_layers:
            loss += beta * style_loss[compt_style_layer]
            grad += beta * style_grad[compt_style_layer]
            compt_style_layer -= 1

        if layer == structure_layer:
            loss += alpha * structure_loss
            grad += alpha * structure_grad

        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    return loss, grad


def compute_inverse_loss(o_features, p_features, img, net, layer):
        structure_loss, structure_grad = structure_error(p_features, \
                        o_features, mode = "inverse")
        structure_loss /= norm(p_features) ** 2
        structure_grad /= norm(p_features) ** 2

        norm_loss, norm_grad = alpha_reg(img, alpha = 6, lambd = 1)
        TV_loss, TV_grad = TV_reg(img, beta = 1, lambd = 1)

        loss = structure_loss
        grad = structure_grad

        net.blobs[layers[-1]].diff[:] = grad
        net.backward(start=layer, end='data')

        grad = net.blobs["data"].diff[0] + norm_grad + TV_grad
        loss += norm_loss + TV_loss
        return loss, grad


def minimize(fun, x0, grad):
    row, col = x0.shape
    bnds = [(0, 255) for i in np.array(row*col)]

    res = minimize(loss, output_transformed, method = "L-BFGS-B", \
                   jac = grad, bounds = bnds)
    out = res.x

    return out
