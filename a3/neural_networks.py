import tensorflow as tf

# Q1.1.1 layer-wise building block
def create_new_layer(input_tensor, num_hidden_units):
    '''
        @param input_tensor - outputs of the previous layer in the neural network, without the bias term.
        @param num_hidden_units - number of hidden units to use for this new layer
    '''
    # Create the new layer weight matrix using Xavier initialization
    input_dim = int(input_tensor.shape[0])
    initializer = tf.contrib.layers.xavier_initializer()
    W_shape = [input_dim, num_hidden_units]
    W = tf.Variable(initializer(W_shape))
    b = tf.Variable(tf.zeros([num_hidden_units, 1]), dtype=tf.float32)

    # MatMul the extended input tensor by the new weight matrix and add the biases
    output_tensor = tf.matmul(W, input_tensor, True) + b

    # Return this operation
    return output_tensor
    
with tf.Session() as sess:
    t = tf.constant([[1.],[2.],[3.]])
    create_new_layer(t, 3)
    create_new_layer(t, 4)
