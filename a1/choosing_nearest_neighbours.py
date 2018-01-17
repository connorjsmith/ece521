from euclidean_distance import euclidean_distance
import numpy as np
import tensorflow as tf

# Returns the responsibility vector r* for the new_point
def get_responsibility_vector_indices(training_vectors, new_point, k):
    new_point = new_point.reshape(1, new_point.shape[0]) # 1 x D matrix to match euclidean_distance function
    pairwise_distances = euclidean_distance(training_vectors, new_point)
    _, indices = tf.nn.top_k(-pairwise_distances.flatten(), k, sorted=True) # k nearest points
    return indices.eval()

def get_responsibility_vector(training_vectors, new_point, k):
    indices = get_responsibility_vector_indices(training_vectors, new_point, k)
    responsibility_value = 1./k
    r = np.zeros(training_vectors.shape[0])
    r[indices] = responsibility_value # sets the value of all elements at a position listed in `indices` to be set to 1/k
    return r # yT . r = k-NN prediction function y^

