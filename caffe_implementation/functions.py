from numpy.linalg import norm
import numpy as np
import caffe

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
    A = tf.squeeze(A)
    row, col, channel = A.shape
    Gram = np.matrix(np.zeros((row, col)))

    for i in np.arange(channel):
        for j in np.arange(i, channel, 1):
            # computes correlation between response of filter i and j
            product = A[:,:,i]*A[:,:,j]
            Gram[i,j] = np.sum(product)

    Gram = Gram + Gram.T - np.diag(Gram.diagonal())

    return Gram


def style_error(features_a, features_x, w = np.ones(5)/5):
    """
    Computes style error based on the definition of the paper.
    w is a vector of weights of size (1,5)
    """
    E = 0
    row, col, channel = features_a[k].shape

    for k in np.arange(5):
        N_I = channel
        M_I = row * col

        A = style_representation(features_a[k])
        G = style_representation(features_x[k])

        grad = A - G
        E += w[k] / (4* M_I**2 * N_I**2) * norm(grad) ** 2
        grad = grad * (G > 0)

    return E


def structure_error(features_p, features_x, layer_index):
    """
    Computes structure_error based on the definiton of the paper.
    """
    E = 0
    row, col, channel = features_a[k].shape

    P = features_p[layer_index]
    F = features_x[layer_index]

    N_I = channel
    M_I = row * col

    P = tf.reshape(P, [M_I, N_I])
    F = tf.reshape(F, [M_I, N_I])

    grad = P-F
    E = 0.5 * norm(grad) ** 2
    grad = grad * (F > 0)

    return E


def alpha_reg(x, alpha, lambd):
    """
    Computes alpha-norm-regularisation term with weight lambd
    """
    (row, col, channel)  = (int(x.get_shape()[1]), int(x.get_shape()[2]), \
                            int(x.get_shape()[3]))
    norms = []
    norm_a = tf.zeros([1])

    for k in np.arange(channel):
        xt = tf.reshape(x[:,:,:,0], [row*col, 1])
        x_mean = tf.reduce_mean(xt)
        x_c = tf.sub(xt, x_mean * tf.ones_like(xt))
        powers = alpha * tf.ones_like(xt)
        norms.append(tf.reduce_sum(tf.pow(x_c, powers)))

    norm_a = tf.add(tf.add(norms[0], norms[1]), norms[2])

    return lambd * norm_a


def TV_reg(x, beta, lambd):
    """
    Computes TV-regularisation term with power beta/2 and weight lambd
    """
    (row, col, channel)  = (int(x.get_shape()[1]), int(x.get_shape()[2]), \
                            int(x.get_shape()[3]))
    norms = []
    norm_b = tf.zeros([1])
    for k in np.arange(channel):
        xt = x[:,:,:,k]
        gx = tf.sub(xt[:, :,1:], xt[:, :,:-1])
        gy = tf.sub(xt[:, 1:,:], xt[:, :-1,:])

        # makes gx and gy square tensors anew
        gx2 = tf.concat(2, [gx, tf.zeros([1, row, 1])])
        gy2 = tf.concat(1, [gy, tf.zeros([1, 1, col])])

        beta2 = np.float(beta)/2
        power2 = 2*tf.ones_like(xt)
        powerbeta = beta2*tf.ones_like(xt)
        loss_k = tf.reduce_sum(tf.pow(tf.pow(gx2, 2) \
                                + tf.pow(gy2, 2), powerbeta))
        norms.append(loss_k)

    norm_b = tf.add(tf.add(norms[0], norms[1]), norms[2])

    return lambd * norm_b
