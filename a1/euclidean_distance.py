import tensorflow as tf


def euclidean_distance(X, Z):
    D = X.shape[-1]
    X_int = tf.reshape(X, [-1, 1, D])
    Z_int = tf.reshape(Z, [1, -1, D])

    distance_pairs = X_int - Z_int
    eucl_dist = tf.reduce_sum(tf.square(distance_pairs), -1, name="euclidean_distances")
    return eucl_dist

if __name__ == '__main__':
    session = tf.InteractiveSession()
    X = tf.constant([[1,2,3], [4,5,6]])
    Z = tf.constant([[7,8,9], [1,2,3]])

    expected_result = tf.constant([[108, 0], [27, 27]])

    tf.assert_equal(euclidean_distance(X,Z), expected_result)
    print(session.run(euclidean_distance(X,Z)))
