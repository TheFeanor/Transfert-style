from numpy.linalg import norm


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
    shape_A = A.shape
    Gram = np.zeros(shape_A[1], shape_A[2])

    for i in np.arange(shape_A[3]):
        for j in np.arange(shape_A[3]):
            # computes correlation between response of filter i and j
            product = A[:,:,:,i] * A[:,:,:,j]
            Gram[i,j] = np.sum(product)

    return Gram


def style_error(features_a, features_x, w = np.ones(5)/5):
    """
    Computes style error based on the definition of the paper.
    w is a vector of weights of size (1,5)
    """
    E = 0

    for k in np.arange(5):
        N_I = features_a[k].shape[3]
        M_I = features_a[k].shape[1] * features_a[k].shape[2]

        A = style_representation(features_a[k])
        G = style_representation(features_x[k])

        E += w[k] / (4* M_I**2 * N_I**2) * norm(A-G)**2

    return E
