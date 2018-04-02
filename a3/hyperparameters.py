import tensorflow as tf
import neural_networks.py as nns
import util.py as ut

def create_new_layer(input_tensor, num_hidden_units):
    '''
        @param input_tensor - outputs of the previous layer in the neural network, without the bias term.
        @param num_hidden_units - number of hidden units to use for this new layer
    '''
    # Create the new layer weight matrix using Xavier initialization
    input_dim = int(input_tensor.shape[-1])
    initializer = tf.contrib.layers.xavier_initializer()
    W_shape = [input_dim, num_hidden_units]
    W = tf.get_variable("W", initializer=initializer(W_shape), dtype=tf.float32)
    # todo: zero initializer?
    b = tf.get_variable("b", shape=[1, num_hidden_units], dtype=tf.float32)

    # MatMul the extended input tensor by the new weight matrix and add the biases
    output_tensor = tf.matmul(input_tensor, W) + b

    # Return this operation
    return output_tensor

def number_of_hidden_units():
    '''
    '''
    # Constants
    decay = 0.0003
    B = 500
    iters = 10000
    lr = 0.005
    hidden_units = [100,500,1000]
    
    # Load data
    (trainData, trainTarget, validData, validTarget,
         testData, testTarget) = ut.load_notMNIST()
    
    # Precalculations
    num_iters_per_epoch = len(trainData)//B # number of iterations we have to do for one epoch
    print("Num epochs = ",iters/num_iters_per_epoch)
	
    Xslice, Yslice = tf.train.slice_input_producer([trainData, trainTarget], num_epochs=None)
    Xbatch, Ybatch = tf.train.batch([Xslice, Yslice], batch_size = B)
	
    # Set place-holders & variables
    X = tf.placeholder(tf.float64, shape=(None, trainData.shape[0]), name='X')
    Y = tf.placeholder(tf.float64, shape=(None, 10), name='Y')
	
    for h in range(0, len(hidden_units)):
        
        # Build graph
        with tf.variable_scope("layer1"):
            s_1 = create_new_layer(X, hidden_units(h))
        x_1 = tf.nn.relu(s_1)
        with tf.variable_scope("layer2"):
            s_2 = create_new_layer(x_1, 10)
        x_2 = tf.nn.softmax(s_2)
		
        # Calculate loss & accuracy
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_2, labels=Y))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x_2, 1), tf.argmax(Y, 1)), tf.float32))
        
        print("Number of hidden units", h)

        with tf.Session() as sess:
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(iters):
                sess.run([optimizer], feed_dict={X: Xbatch, Y: Ybatch})
                if (i % num_iters_per_epoch == 0):
                    t_loss, t_acc = sess.run([loss, accuracy], feed_dict={X: trainData, Y: trainTarget})
                    v_loss, v_acc = sess.run([loss, accuracy], feed_dict={X: validData, Y: validTarget})
                    test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X: testData, Y: testTarget})
                    print("Epoch: {}, Training Loss: {}, Accuracies: [{}, {}, {}]".format(i//num_iters_per_epoch, t_loss, t_acc, v_acc, test_acc))
        