def tensor_norm(A, is_square_norm):
    squareA = tf.sqare(A)
    normA = tf.sum(squareA)

    if is_square_norm:
        out = tf.square(normA)
    else:
        out = normA

    return out


def tensor_distance(A, B, is_square_norm):
    substraction = tf.sub(A, B)
    distance = tensor_norm(substraction, is_square_norm=)

    return distance
