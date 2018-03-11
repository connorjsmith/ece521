import tensorflow as tf
import matplotlib.pyplot as plt
import util

def plot_data(data_arr, name):
    plt.figure(figsize=(8, 6), dpi=80)

    plt.subplot(1, 1, 1)

    epoch = 1
    for y in data_arr:
        plt.plot(y, linewidth=1.0, linestyle='-', label=r'$\eta$' + " = " + str(epoch))
        epoch += 1

    plt.xlabel("# of Epochs")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.savefig(name, format="pdf")


def binary_cross_entropy():
    xTrain, yTrain, xValid, yValid, xTest, yTest = util.load_data()

    with tf.Graph().as_default():
        decay = 0.01
        B = 500
        learning_rates = [0.001]
        iters = 20000
        num_iters_per_epoch = len(xTrain)//B # number of iterations we have to do for one epoch
        print("Num epochs = ",iters/num_iters_per_epoch)

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

        rate_loss_dict = dict()
        for r in learning_rates:
            loss_amounts = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(iters):
                    # run one iteration of the optimizer
                    sess.run([optimizer], feed_dict={learning_rate: r})
                    # calculate our loss for this iteration
                    if (i % num_iters_per_epoch == 0):
                        loss_amount = sess.run(loss)
                        print("Epoch {}, loss = {}".format(i//num_iters_per_epoch, loss_amount))
                        loss_amounts.append(loss_amount)
                coord.request_stop()
                coord.join(threads)
            plot_data(loss_amounts, "log_loss_str(r)")
            rate_loss_dict[r] = loss_amounts



plot_data([], "test")
binary_cross_entropy()

