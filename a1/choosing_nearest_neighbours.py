from euclidean_distance import euclidean_distance
import numpy as np
import tensorflow as tf

# Returns the responsibility vector r* for the new_point
def get_responsibility_matrix_indices(training_vectors, new_points, k):
    new_points = tf.reshape(new_points, [1,-1])
    pairwise_distances = euclidean_distance(training_vectors, new_points)
    values, indices = tf.nn.top_k(tf.transpose(-pairwise_distances), k, sorted=True, name="responsibility_indices") # k nearest points
    return indices


def get_responsibility_matrix(training_vectors, new_points, k):
    indices = get_responsibility_matrix_indices(training_vectors, new_points, k)
    responsibility_value = 1/k
    off_value = tf.constant(0, dtype=tf.float64)
    r_depth = training_vectors.shape[0]
    r = tf.one_hot(indices=tf.transpose(indices), depth=r_depth, on_value=responsibility_value, off_value=off_value, dtype=tf.float64)
    r = tf.reduce_sum(r, axis=0)
    return r # yT . r = k-NN prediction function y^


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    t = tf.constant([[1,2,3],[4,5,6], [7,1,3], [6,0,1], [7,8,9], [3,6,8]])
    n = tf.constant([[3,4,5], [7,2,4], [3,4,7], [6,0,3]])
    k = tf.constant(3, dtype=tf.int32)

    # tf.assert_equal(get_responsibility_matrix_indices(t,n,k), expected_indices)
    print(sess.run(get_responsibility_matrix_indices(t,n,k)))

    print(sess.run(get_responsibility_matrix(t,n,k)))

