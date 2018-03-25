# Q1.1.1 layer-wise building block
def create_new_layer(input_tensor, num_hidden_units):
    '''
        @param input_tensor - outputs of the previous layer in the neural network, without the bias term.
        @param num_hidden_units - number of hidden units to use for this new layer
    '''
    # Create the new layer weight matrix using Xavier initialization
    input_dim = input_tensor.shape[-1]
    W = tf.Variable(shape=[input_dim, num_hidden_units], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(num_hidden_units))

    # MatMul the extended input tensor by the new weight matrix and add the biases
    output_tensor = tf.matmul(W, input_tensor_padded) + b

    # Return this operation
    return output_tensor
    