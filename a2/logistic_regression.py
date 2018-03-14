import tensorflow as tf
import numpy as np
import util

def binary_cross_entropy(Q=1):
    xTrain, yTrain, xValid, yValid, xTest, yTest = util.load_data()

    with tf.Graph().as_default():
        decay = 0
        B = 500
        learning_rates = [0.005, 0.001, 0.0001]
        iters = 5000
        num_iters_per_epoch = len(xTrain)//B # number of iterations we have to do for one epoch
        print("Num epochs = ",iters/num_iters_per_epoch)

        # optimized parameters
        w = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5, seed=521), dtype=tf.float32, name="weight-vector")
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

        # setting up batch loss function
        y_pred = tf.matmul(Xbatch, w) + b
        logLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=ybatch)) + decay * tf.nn.l2_loss(w)

        # setting up epoch loss function
        train_logLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.matmul(xTrainTensor, w)+b, labels=yTrainTensor)) + decay * tf.nn.l2_loss(w)

        # accuracy function
        train_y_pred = tf.round(tf.sigmoid(tf.matmul(xTrainTensor, w) + b))
        valid_y_pred = tf.round(tf.sigmoid(tf.matmul(xValidTensor, w) + b))
        train_accuracy = tf.count_nonzero(tf.equal(train_y_pred, yTrainTensor)) / yTrainTensor.shape[0]
        valid_accuracy = tf.count_nonzero(tf.equal(valid_y_pred, yValidTensor)) / yValidTensor.shape[0]

        # optimizer function
        gradientOptimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(logLoss)

        if Q==2: # Part 2.1 Q2
            adamOptimizer = tf.train.AdamOptimizer(learning_rate).minimize(logLoss)
            for optimizer, name in [(gradientOptimizer, "Gradient Descent"), (adamOptimizer, "Adam Optimizer")]:
                loss_amounts = []
                valid_accuracies = []
                train_accuracies = []
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    print("Running", name)
                    for i in range(iters):
                        sess.run([optimizer], feed_dict={learning_rate: 0.001})
                        if (i % num_iters_per_epoch == 0):
                            loss_amount, train_acc, valid_acc = sess.run([train_logLoss, train_accuracy, valid_accuracy])
                            loss_amounts.append(loss_amount)
                            valid_accuracies.append(valid_acc)
                            train_accuracies.append(train_acc)
                            print("Epoch: {}, Loss: {}".format(i//num_iters_per_epoch, loss_amount))
                    coord.request_stop()
                    coord.join(threads)
                    np.save("{}_loss".format(name),loss_amounts)
                    np.save("{}_v_acc".format(name), valid_accuracies)
                    np.save("{}_t_acc".format(name), train_accuracies)
        elif Q==1: # Part 2.1 Q1
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
                        sess.run([gradientOptimizer], feed_dict={learning_rate: r})
                        # calculate our loss for this iteration
                        if (i % num_iters_per_epoch == 0):
                            loss_amount, train_acc, valid_acc = sess.run([train_logLoss, train_accuracy, valid_accuracy])
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
        elif Q==3: # Part 2.1 Q3
            loss_amounts = []
            valid_accuracies = []
            train_accuracies = []
            logOptimizer = tf.train.AdamOptimizer(learning_rate).minimize(logLoss)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(iters):
                    # run one iteration of the optimizer
                    sess.run([logOptimizer], feed_dict={learning_rate: 0.001})
                    # calculate our loss for this iteration
                    if (i % num_iters_per_epoch == 0):
                        loss_amount, train_acc, valid_acc = sess.run([train_logLoss, train_accuracy, valid_accuracy])
                        print("Epoch {}, loss = {}".format(i//num_iters_per_epoch, loss_amount))
                        print("\t Train Acc = {}, Valid Acc = {}".format(train_acc, valid_acc))
                        loss_amounts.append(loss_amount)
                        valid_accuracies.append(valid_acc)
                        train_accuracies.append(train_acc)

                coord.request_stop()
                coord.join(threads)
                np.save("2.1.3_Log_loss",loss_amounts)
                np.save("2.1.3_Log_v_acc", valid_accuracies)
                np.save("2.1.3_Log_t_acc", train_accuracies)

            loss_amounts = []
            valid_accuracies = []
            train_accuracies = []
            lin_t_y_pred = tf.minimum(tf.maximum(tf.ceil(tf.matmul(xTrainTensor, w) + b), 0), 1)
            lin_v_y_pred = tf.minimum(tf.maximum(tf.ceil(tf.matmul(xValidTensor, w) + b), 0), 1)
            linearLoss = tf.reduce_mean(tf.square(y_pred - ybatch))/2
            linearLoss_epoch = tf.reduce_mean(tf.square(tf.matmul(xTrainTensor, w) + b - yTrainTensor))/2
            lin_accuracy = tf.count_nonzero(tf.equal(tf.minimum(tf.maximum(tf.ceil(y_pred), 0), 1), ybatch)) / yTrainTensor.shape[0]
            lin_t_accuracy = tf.count_nonzero(tf.equal(lin_t_y_pred, yTrainTensor)) / yTrainTensor.shape[0]
            lin_v_accuracy = tf.count_nonzero(tf.equal(lin_v_y_pred, yValidTensor)) / yValidTensor.shape[0]
            linearOptimizer = tf.train.AdamOptimizer(learning_rate).minimize(linearLoss)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for i in range(iters):
                    # run one iteration of the optimizer
                    sess.run([linearOptimizer], feed_dict={learning_rate: 0.001})
                    # calculate our loss for this iteration
                    if (i % num_iters_per_epoch == 0):
                        loss_amount, train_acc, valid_acc = sess.run([linearLoss_epoch, lin_t_accuracy, lin_v_accuracy])
                        print("Epoch {}, loss = {}".format(i//num_iters_per_epoch, loss_amount))
                        print("\t Train Acc = {}, Valid Acc = {}".format(train_acc, valid_acc))
                        loss_amounts.append(loss_amount)
                        valid_accuracies.append(valid_acc)
                        train_accuracies.append(train_acc)
                np.save("2.1.3_Lin_loss",loss_amounts)
                np.save("2.1.3_Lin_v_acc", valid_accuracies)
                np.save("2.1.3_Lin_t_acc", train_accuracies)

                coord.request_stop()
                coord.join(threads)
                
            

binary_cross_entropy(3)
