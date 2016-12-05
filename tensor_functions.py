from numpy.linalg import norm
import numpy as np
import tensorflow as tf

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
    channel = A.get_shape()[2]
    Gram = np.zeros((channel, channel))

    for i in np.arange(channel):
        for j in np.arange(channel):
            # computes correlation between response of filter i and j
            product = tf.mul(A[:,:,i], A[:,:,j])
            Gram[i,j] = tf.reduce_sum(product)

    return Gram


def style_error(features_a, features_x, w = np.ones(5)/5):
    """
    Computes style error based on the definition of the paper.
    w is a vector of weights of size (1,5)
    """
    E = 0

    for k in np.arange(5):
        N_I = features_a[k].get_shape()[3]
        M_I = features_a[k].get_shape()[1] * features_a[k].get_shape()[2]

        A = style_representation(features_a[k])
        G = style_representation(features_x[k])

        E += tf.reduce_sum(w[k] / (4* M_I**2 * N_I**2) * \
                           tf.squared_difference(A, G))

    return E


def structure_error(features_p, features_x, layer_index):
    """
    Computes structure_error based on the definiton of the paper.
    """
    E = 0

    P = features_p[layer_index]
    F = features_x[layer_index]

    N_I = P.get_shape()[3]
    M_I = P.get_shape()[1] * P.get_shape()[2]

    P = tf.reshape(P, [M_I, N_I])
    F = tf.reshape(F, [M_I, N_I])

    E = 0.5 * tf.reduce_sum(tf.squared_difference(P, F))

    return E


def alpha_reg(x, alpha, lambd):
    """
    Computes alpha-norm-regularisation term with weight lambd
    """
    x = x.reshape((224*224,3))
    norm_a = np.power(norm(x, ord=alpha), alpha)

    return lambd * norm_a


def TV_reg(x, beta, lambd):
    """
    Computes TV-regularisation term with power beta/2 and weight lambd
    """
    grad = np.gradient(x)
    gx = grad[0]
    gy = grad[1]
    beta2 = np.float(beta)/2
    norm_b = np.sum(np.power(gx**2 + gy**2, beta2))

    return lambd * norm_b
