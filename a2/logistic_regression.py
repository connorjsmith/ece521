import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import util

def plot_data_learning(data_vector, names, pdf_name):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)

    for (y, name) in zip(data_vector, names):
        plt.plot(y, linewidth=1.0, linestyle='-', label=name)

    plt.suptitle("Cross-Entropy Loss vs. Number of Epochs (Part 2.1 Q1)", fontsize=14, y=0.97)
    plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
    plt.xlabel("# of Epochs")
    plt.ylabel("Training Loss (Cross-Entropy Loss)")
    plt.legend(title="Training Rate (η)", loc="upper right")
    plt.grid('on', linestyle='-', linewidth=0.5)
    plt.savefig(pdf_name, format="pdf")


def plot_data_plain_SGD(data_vector, names, pdf_name):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)

    for (y, name) in zip(data_vector, names):
        plt.plot(y, linewidth=1.0, linestyle='-', label=name)

    plt.suptitle("Cross-Entropy Loss vs. Number of Epochs (Part 2.1 Q1)", fontsize=14, y=0.97)
    plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
    plt.xlabel("# of Epochs")
    plt.ylabel("Training Loss (Cross-Entropy Loss)")
    plt.legend(title="Training Rate (η)", loc="upper right")
    plt.grid('on', linestyle='-', linewidth=0.5)
    plt.savefig(pdf_name, format="pdf")


def plot_data_lin_regr_comparison(data_vector, names, pdf_name):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)

    for (y, name) in zip(data_vector, names):
        plt.plot(y, linewidth=1.0, linestyle='-', label=name)

    plt.suptitle("Cross-Entropy Loss vs. Number of Epochs (Part 2.1 Q1)", fontsize=14, y=0.97)
    plt.title("Weight-Decay λ=0.01, Mini-Batch Size B=500", fontsize=10)
    plt.xlabel("# of Epochs")
    plt.ylabel("Training Loss (Cross-Entropy Loss)")
    plt.legend(title="Training Rate (η)", loc="upper right")
    plt.grid('on', linestyle='-', linewidth=0.5)
    plt.savefig(pdf_name, format="pdf")


def binary_cross_entropy():
    xTrain, yTrain, xValid, yValid, xTest, yTest = util.load_data()

    with tf.Graph().as_default():
        decay = 0.01
        B = 500
        learning_rates = [0.005, 0.001, 0.0001]
        iters = 5000
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
        xValidTensor = tf.constant(xValid, dtype=tf.float32, name="X-Validation")
        yValidTensor = tf.constant(yValid, dtype=tf.float32, name="Y-Validation")

        Xslice, yslice = tf.train.slice_input_producer([xTrainTensor, yTrainTensor], num_epochs=None)

        Xbatch, ybatch = tf.train.batch([Xslice, yslice], batch_size = B)

        # setting up loss function
        y_pred = tf.matmul(Xbatch, w) + b
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=ybatch)) + decay/2 * tf.nn.l2_loss(w)

        # accuracy function
        train_y_pred = tf.matmul(xTrainTensor, w) + b
        valid_y_pred = tf.matmul(xValidTensor, w) + b
        train_accuracy = 1 - tf.count_nonzero(tf.equal(tf.round(train_y_pred), yTrainTensor)) / yTrainTensor.shape[0]
        valid_accuracy = 1 - tf.count_nonzero(tf.equal(tf.round(valid_y_pred), yValidTensor)) / yValidTensor.shape[0]

        # optimizer function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        for r in learning_rates:
            loss_amounts = []
            valid_accuracies = []
            train_accuracies = []
            print("Running learning rate", r)
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
                        loss_amount, train_acc, valid_acc = sess.run([loss, train_accuracy, valid_accuracy])
                        print("Epoch {}, loss = {}".format(i//num_iters_per_epoch, loss_amount))
                        print("\t Train Acc = {}, Valid Acc = {}".format(train_acc, valid_acc))
                        loss_amounts.append(loss_amount)
                        valid_accuracies.append(valid_acc)
                        train_accuracies.append(train_acc)

                coord.request_stop()
                coord.join(threads)
                np.save("log_q1_"+str(r)+"_loss.npy", loss_amounts)
                np.save("log_q1_"+str(r)+"_valid_acc.npy", valid_accuracies)
                np.save("log_q1_"+str(r)+"_train_acc.npy", train_accuracies)



binary_cross_entropy()

