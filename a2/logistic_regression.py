import tensorflow as tf
import util


def binary_cross_entropy():
    xTrain, yTrain, xValid, yValid, xTest, yTest = util.load_data()

    with tf.Graph().as_default():
        decay = 0.01
        B = 500
        learning_rates = [0.001]
        iters = 100

        # optimized parameters
        w = tf.Variable(tf.zeros([784,1]), dtype=tf.float32, name="weight-vector")
        b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias-term")

        # hyperparameters
        learning_rate = tf.placeholder(dtype=tf.float32, name="learning-rate")

        # Get Data
        xTrainTensor = tf.constant(xTrain, dtype=tf.float32, name="X-Training")
        yTrainTensor = tf.constant(yTrain, dtype=tf.float32, name="Y-Training")

        Xslice, yslice = tf.train.slice_input_producer([xTrainTensor, yTrainTensor], num_epochs=None)

        Xbatch, ybatch = tf.train.batch([Xslice, yslice], batch_size = B)

        # setting up loss function
        y_pred = tf.matmul(Xbatch, w) + b
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=ybatch)) + decay/2 * tf.nn.l2_loss(w)

        # optimizer function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        for r in learning_rates:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(iters):
                    print("iteration", i)
                    # run one iteration of the optimizer
                    sess.run([optimizer], feed_dict={learning_rate: r})
                    # calculate our loss for this iteration
                    loss_amount = sess.run(loss)
                    print("loss =", loss_amount)
                coord.request_stop()
                coord.join(threads)



binary_cross_entropy()
    
