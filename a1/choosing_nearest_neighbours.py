from euclidean_distance import euclidean_distance
import numpy as np
import tensorflow as tf

# Returns the responsibility vector r* for the new_point
def get_responsibility_vector_indices(training_vectors, new_point, k):
    new_point = tf.reshape(new_point, [1,-1])
    pairwise_distances = euclidean_distance(training_vectors, new_point)
    values, indices = tf.nn.top_k(tf.reshape(-pairwise_distances, [-1]), k, sorted=True, name="responsibility_indices") # k nearest points
    return indices


def get_responsibility_vector(training_vectors, new_point, k):
    indices = get_responsibility_vector_indices(training_vectors, new_point, k)
    responsibility_value = 1/k
    off_value = tf.constant(0, dtype=tf.float64)
    r_depth = training_vectors.shape[0]
    r = tf.one_hot(indices=indices, depth=r_depth, on_value=responsibility_value, off_value=off_value, dtype=tf.float64)
    r = tf.reduce_sum(r, axis=0)
    return r # yT . r = k-NN prediction function y^


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    t = tf.constant([[1,2,3],[4,5,6], [7,1,3], [6,0,1], [7,8,9], [3,6,8]])
    n = tf.constant([[3,4,5]])
    expected_indices = tf.constant([[1,0,3]])
    k = tf.constant(3, dtype=tf.int32)

    # tf.assert_equal(get_responsibility_vector_indices(t,n,k), expected_indices)
    print(sess.run(get_responsibility_vector_indices(t,n,k)))

    print(sess.run(get_responsibility_vector(t,n,k)))

