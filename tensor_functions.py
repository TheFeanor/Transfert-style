def tensor_norm(A, is_square_norm):
    """
    Computes de L2-norm (Euclian norm) for a feature map A.
    The parameters is_square_norm denotes wether the output is the norm
    or the square norm.
    """
    squareA = tf.sqare(A)
    normA = tf.sum(squareA)

    if is_square_norm:
        out = tf.square(normA)
    else:
        out = normA

    return out


def tensor_distance(A, B, is_square_norm):
    """
    Computes the Euclidian distance between two features maps A and B.
    The parameters is_square_norm denotes wether the output is the distance
    or the square distance.
    """
    substraction = tf.sub(A, B)
    distance = tensor_norm(substraction, is_square_norm)

    return distance


def response_correlation(A, B):
    """
    Computes de Gram matrix correlation between two filter responses A and B
    """
    product = tf.mul(A, B)
    gram = tf.sum(product)

    return gram


def style_representation(A):
    """
    Computes the style representation of the feature map A.
    """

    shape = tf.get_shape(A)
    Gram = np.zeros(shape[1:3])

    for i in np.arange(shape[1]):
        for j in np.arange(shape[2]):
            # computes correlation between response of filter i and j
            Gram[i,j] = response_correlation(A[i,:,:,:], A[j,:,:,:])

    return tf.convert_to_tensor(Gram, dtype=tf.float32)
