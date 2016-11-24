def tensor_norm(A, is_square_norm):
    squareA = tf.sqare(A)
    normA = tf.sum(squareA)

    if is_square_norm:
        out = tf.square(normA)
    else:
        out = normA

    with tf.Session() as sess:
        result = tf.run(out)

    return result


def tensor_distance(A, B, is_square_norm):
    substraction = tf.sub(A, B)
    distance = tensor_norm(substraction, is_square_norm=)

    with tf.Session() as sess:
        result = tf.run(distance)

    return result
