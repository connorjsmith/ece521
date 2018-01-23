# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:51:29 2018

@author: Jeffrey
"""

import tensorflow as tf
from euclidean_distance import euclidean_distance

def kNN_classification(test_point, in_features, targets, k):
    distances = euclidean_distance(test_point, in_features)
    (val, ind) = tf.nn.top_k(-distances,k) # find closest neighbours in training set
    
    candidates = tf.gather(tf.constant(targets),ind) # Find the classifications for these neighbours
    length = tf.shape(candidates)[0]
    class_list = []
    count_list = []

    # Count the frequency of nearest neighbours and put them into matrices
    # (reduced class list and count list)
    for i in range(0,length.eval()):
        (temp_class, __, temp_count) = tf.unique_with_counts(candidates[i])
        padding = tf.concat([tf.constant([0]), tf.constant([k]) - tf.shape(temp_class)],0)
        class_list.append(tf.pad(temp_class,[padding]))
        count_list.append(tf.pad(temp_count,[padding]))
    
    red_class_list = tf.stack(class_list)
    red_count_list = tf.stack(count_list)
    
    # Create an iterator for each test_point
    iterator = tf.cast(tf.linspace(0., length.eval() - 1., length.eval()), tf.int64)
    iterator = tf.reshape(iterator,[length,1])

    # Combine the iterator with the indices of the highest counts in the
    # reduced count list (red_count_list)
    count_loc = tf.concat([iterator, tf.reshape(tf.argmax(red_count_list,1),[length,1])],1)
    
    outputs = tf.gather_nd(red_class_list, count_loc)

    return outputs

# Takes in 2 vectors and returns the % of occurances they are the same elementwise
def classification_performance(results, targets): 
    error = tf.count_nonzero(results - targets) / tf.cast(tf.shape(targets), tf.int64)
    return tf.cast(tf.constant(1.), tf.float64) - error