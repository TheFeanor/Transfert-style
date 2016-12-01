from numpy.linalg import norm


def features_distance(A, B):
    """
    Computes the Euclidian distance between two features maps A and B.
    The parameters is_square_norm denotes wether the output is the distance
    or the square distance.
    """
    distance = norm(A-B)**2

    return distance


def responses_correlation(A, B):
    """
    Computes de Gram matrix correlation between two filter responses A and B
    """
    product = A*B
    gram = np.sum(product)

    return gram


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
            Gram[i,j] = responses_correlation(A[:,:,:,i], A[:,:,:,j])

    return Gram
