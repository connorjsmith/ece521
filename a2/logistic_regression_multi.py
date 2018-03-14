import tensorflow as tf
import numpy as np
import util

def load_notmnist_data():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        t = np.zeros((trainTarget.shape[0], 10))
        t[np.arange(trainTarget.shape[0]), trainTarget] = 1
        trainTarget = t
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        t = np.zeros((validTarget.shape[0], 10))
        t[np.arange(validTarget.shape[0]), validTarget] = 1
        validTarget = t
        testData, testTarget = Data[16000:], Target[16000:]
        t = np.zeros((testTarget.shape[0], 10))
        t[np.arange(testTarget.shape[0]), testTarget] = 1
        testTarget = t
        return (trainData.reshape(trainData.shape[0], -1), trainTarget, validData.reshape(validData.shape[0], -1), validTarget, testData.reshape(testData.shape[0], -1), testTarget)
        

def multiclass_not_mnist():
    xTrain, yTrain, xValid, yValid, xTest, yTest = load_notmnist_data()
    with tf.Graph().as_default():
        decay = 0.01
        B = 500
        iters = 5000
        learning_rate = tf.constant(0.001, dtype=tf.float32, name="learning-rate")
        

        num_iters_per_epoch = len(xTrain)//B # number of iterations we have to do for one epoch
        print("Num epochs = ",iters/num_iters_per_epoch)

        # optimized parameters
        w = tf.Variable(tf.truncated_normal(shape=[784,10], stddev=0.5, seed=521), dtype=tf.float32, name="weight-vector")
        b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias-term")

        # input tensors
        xTrainTensor = tf.constant(xTrain, dtype=tf.float32, name="X-Training")
        yTrainTensor = tf.constant(yTrain, dtype=tf.float32, name="Y-Training")
        xValidTensor = tf.constant(xValid, dtype=tf.float32, name="X-Validation")
        yValidTensor = tf.constant(yValid, dtype=tf.float32, name="Y-Validation")

        # Create randomly shuffled batches 
        Xslice, yslice = tf.train.slice_input_producer([xTrainTensor, yTrainTensor], num_epochs=None)

        Xbatch, ybatch = tf.train.batch([Xslice, yslice], batch_size = B)

        # setting up batch loss function
        y_pred = tf.matmul(Xbatch, w) + b
        softmaxLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=ybatch)) + decay * tf.nn.l2_loss(w)

        # accuracy function TODO these are also probably wrong
        train_y_pred = tf.sigmoid(tf.matmul(xTrainTensor, w) + b)
        valid_y_pred = tf.sigmoid(tf.matmul(xValidTensor, w) + b)
        train_accuracy = tf.count_nonzero(tf.equal(tf.argmax(train_y_pred, 1), tf.argmax(yTrainTensor, 1))) / yTrainTensor.shape[0]
        valid_accuracy = tf.count_nonzero(tf.equal(tf.argmax(valid_y_pred, 1), tf.argmax(yValidTensor, 1))) / yValidTensor.shape[0]

        # optimizer function
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(softmaxLoss)

        loss_amounts = []
        valid_accuracies = []
        train_accuracies = []
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(iters):
                sess.run([optimizer], feed_dict={learning_rate: 0.001})
                if (i % num_iters_per_epoch == 0):
                    loss_amount, train_acc, valid_acc = sess.run([softmaxLoss, train_accuracy, valid_accuracy])
                    loss_amounts.append(loss_amount)
                    valid_accuracies.append(valid_acc)
                    train_accuracies.append(train_acc)
                    print("Epoch: {}, Loss: {}, trainAcc: {}".format(i//num_iters_per_epoch, loss_amount, train_acc))
            coord.request_stop()
            coord.join(threads)
            np.save("2.2.1_notmnist_loss", loss_amounts)
            np.save("2.2.1_notmnist_v_acc", valid_accuracies)
            np.save("2.2.1_notmnist_t_acc", train_accuracies)


multiclass_not_mnist()
