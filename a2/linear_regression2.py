import _pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def load_data():
    with np.load("./notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255
        Target = Target[dataIndx].reshape(-1,1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]

        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        return (trainData, trainTarget, validData, validTarget, testData, testTarget)


def SGD(xTrain, yTrain, batchSize, iters, learning_rate, decay_coefficient, use_normal_eqn = False):
    """
    Do stochastic gradient descent
    :param xTrain: numpy array of shape (number of samples, width, height)
    :param yTrain: np array of shape (number of samples, 1)
    :param batchSize: integer
    :param iters: integer
    :param learning_rate: float
    :param decay_coefficient: float
    :param use_normal_eqn: bool
    :return: losses for each epoch, list, each element is np.array of shape (1)
    """
    X = tf.placeholder(tf.float64, shape=(None, xTrain.shape[1]*xTrain.shape[2]), name='X')
    Y = tf.placeholder(tf.float64, shape=(None, 1), name='Y')
    with tf.variable_scope('weight', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('weights', shape=[X.shape[1] + 1, 1], dtype=tf.float64)   # (W*H +1) x 1
    x_in = tf.pad(X, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)        # bn x (W*H +1)
    z = tf.tensordot(x_in, weights, axes=1)
    dc = tf.placeholder(tf.float64, name='dc')
    mse_loss = tf.reduce_mean(tf.square(Y-z))/(2)
    decay_loss = dc*tf.sqrt(tf.reduce_sum(tf.square(weights)))/2
    total_loss = mse_loss + decay_loss
    full_mse_loss = tf.reduce_mean(tf.square(Y-tf.tensordot(x_in, weights, axes=1)))/2
    #decay_loss should be average over minibatch as well
    full_total_loss = full_mse_loss + decay_loss
    if use_normal_eqn:
        grads = [tf.reshape(tf.reduce_mean(-tf.transpose((Y - z) * x_in) + tf.expand_dims(dc,axis = 0) * weights, axis=-1), (-1, 1)),]
    else:
        grads = tf.gradients(total_loss, weights)
    updates = tf.assign_sub(weights, learning_rate * grads[0])
    num_sample = xTrain.shape[0]
    num_epochs = (batchSize*iters + 1)//num_sample + 1
    curr_iter = 0
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            inds = np.arange(num_sample)
            np.random.shuffle(inds)
            this_epoch_loss = 0.0
            curr_samples = 0
            for j in range((num_sample+1)//batchSize + 1):
                l = j*batchSize
                u = min(xTrain.shape[0], l+batchSize)
                if l == u:
                    continue
                if curr_iter >= iters:
                    break
                curr_iter += 1
                curr_samples += (u - l)
                this_inds = inds[l:u]
                this_batch_X = xTrain[this_inds]
                this_batch_Y = yTrain[this_inds]
                this_batch_X = np.reshape(this_batch_X, (this_batch_X.shape[0], this_batch_X.shape[1]*this_batch_X.shape[2]))
                this_loss, gr, wts, full_loss = sess.run([total_loss, grads, weights, full_total_loss],
                                        feed_dict={X: this_batch_X, Y:this_batch_Y.astype(np.float64),
                                                   dc:np.array([decay_coefficient], dtype=np.float64)})
                this_epoch_loss += this_loss*(u-l)
                sess.run(updates,
                         feed_dict={X: this_batch_X, Y:this_batch_Y.astype(np.float64),
                                    dc:np.array([decay_coefficient], dtype=np.float64)})
            if curr_samples == 0:
                continue
            this_epoch_loss /= curr_samples
            losses.append(full_loss)
    return [losses, wts]


def tuning_the_learning_rate():

    iters = 20000
    batch_size = 500
    learning_rates = [0.005, 0.001, 0.0001]
    decay_coefficient = 0.

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()

    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)

    for i in range(0,len(learning_rates)):
        loss, _ = SGD(trainData, trainTarget, batch_size, iters, learning_rates[i], decay_coefficient)
        x = np.arange(len(loss))
        plt.plot(x, np.array(loss),label = r'$\eta$' + " = " + str(learning_rates[i]))

    plt.xlim(-200, 3000)
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()
    
    plt.savefig("Tuning the Learning Rate.pdf", format="pdf")
    
    return plt


def effect_of_minibatch_size():
    iters = 20000
    batch_sizes = [500, 1500, 3500]
    learning_rate = 0.005 # TODO get from tuning_the_learning_rate
    decay_coefficient = 0
    losses = []
    
    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    # TODO finish me
    for i in range(0,len(batch_sizes)):
        time_start = time.clock()
        loss = SGD(trainData, trainTarget, batch_sizes[i], iters, learning_rate, decay_coefficient)[0]
        x = np.arange(len(loss))
        plt.plot(x, np.array(loss), label='batch_size = %f' % batch_sizes[i])
        losses.append(loss)
        time_elapsed = (time.clock() - time_start)
        print("Time passed for B = " + str(batch_sizes[i]) + ": " + str(time_elapsed))
        
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()
    return losses


def generalization():
    iters = 20000
    batch_size = 500
    learning_rate = 0.005
    decay_coefficients = [0.0, 0.001, 0.1, 1]

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    final_weights = []
    validation_accuracy = []
    test_accuracy = [];

    for i in range(0,len(decay_coefficients)):
        time_start = time.clock()
        (loss, wt) = SGD(trainData, trainTarget, batch_size, iters, learning_rate, decay_coefficients[i])
        time_elapsed = (time.clock() - time_start)
        print("Time passed for dc = " + str(decay_coefficients[i]) + ": " + str(time_elapsed))
        x = np.arange(len(loss))
        plt.plot(x, np.array(loss), label='decay_coefficient = %f' % decay_coefficients[i])
        final_weights.append(wt)
    
        valid_linear = np.reshape(validData, (validData.shape[0], validData.shape[1]*validData.shape[2]))
        x_in_valid = tf.pad(valid_linear, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)
        valid_y_pred = tf.matmul(x_in_valid, wt)
        same_valid = tf.equal(tf.greater(valid_y_pred, tf.constant(0.5, tf.float64)), tf.constant(validTarget, tf.bool))
        v_accuracy = tf.count_nonzero(same_valid) / tf.constant(validTarget).shape[0]
        with tf.Session() as sess:
            validation_accuracy.append(sess.run(v_accuracy))
        
        test_linear = np.reshape(testData, (testData.shape[0], testData.shape[1]*testData.shape[2]))
        x_in_test = tf.pad(test_linear, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)
        test_y_pred = tf.matmul(x_in_test, wt)
        same_test = tf.equal(tf.greater(test_y_pred, tf.constant(0.5, tf.float64)), tf.constant(testTarget, tf.bool))
        t_accuracy = tf.count_nonzero(same_test) / tf.constant(testTarget).shape[0]
        with tf.Session() as sess:
            test_accuracy.append(sess.run(t_accuracy))
    
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()
    
    return (validation_accuracy, test_accuracy)

def sgd_vs_normal_equation():
    iters = 20000
    batch_size = 500
    learning_rate = 0.005
    decay_coefficient = 0

    (trainData, trainTarget,
     testData, testTarget,
     validData, validTarget) = load_data()
    validation_accuracy = []
    test_accuracy = [];
    
    time_start = time.clock()
    (loss, wt) = SGD(trainData, trainTarget, batch_size, iters, learning_rate, decay_coefficient)
    time_elapsed = (time.clock() - time_start)
    print("Time passed for SGD = " + str(decay_coefficients[i]) + ": " + str(time_elapsed))
    
    valid_linear = np.reshape(validData, (validData.shape[0], validData.shape[1]*validData.shape[2]))
    x_in_valid = tf.pad(valid_linear, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)
    valid_y_pred = tf.matmul(x_in_valid, wt)
    same_valid = tf.equal(tf.greater(valid_y_pred, tf.constant(0.5, tf.float64)), tf.constant(validTarget, tf.bool))
    v_accuracy = tf.count_nonzero(same_valid) / tf.constant(validTarget).shape[0]
    with tf.Session() as sess:
        validation_accuracy.append(sess.run(v_accuracy))
    
    test_linear = np.reshape(testData, (testData.shape[0], testData.shape[1]*testData.shape[2]))
    x_in_test = tf.pad(test_linear, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)
    test_y_pred = tf.matmul(x_in_test, wt)
    same_test = tf.equal(tf.greater(test_y_pred, tf.constant(0.5, tf.float64)), tf.constant(testTarget, tf.bool))
    t_accuracy = tf.count_nonzero(same_test) / tf.constant(testTarget).shape[0]
    with tf.Session() as sess:
        test_accuracy.append(sess.run(t_accuracy))
    
    time_start = time.clock()
    (normal_loss, wt) = SGD(trainData, trainTarget, batch_size, iters, learning_rate, decay_coefficient, True)
    time_elapsed = (time.clock() - time_start)
    print("Time passed for normal = " + str(decay_coefficients[i]) + ": " + str(time_elapsed))
    
    
    valid_linear = np.reshape(validData, (validData.shape[0], validData.shape[1]*validData.shape[2]))
    x_in_valid = tf.pad(valid_linear, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)
    valid_y_pred = tf.matmul(x_in_valid, wt)
    same_valid = tf.equal(tf.greater(valid_y_pred, tf.constant(0.5, tf.float64)), tf.constant(validTarget, tf.bool))
    v_accuracy = tf.count_nonzero(same_valid) / tf.constant(validTarget).shape[0]
    with tf.Session() as sess:
        validation_accuracy.append(sess.run(v_accuracy))
    
    test_linear = np.reshape(testData, (testData.shape[0], testData.shape[1]*testData.shape[2]))
    x_in_test = tf.pad(test_linear, [[0, 0], [0, 1]], "CONSTANT", constant_values=1)
    test_y_pred = tf.matmul(x_in_test, wt)
    same_test = tf.equal(tf.greater(test_y_pred, tf.constant(0.5, tf.float64)), tf.constant(testTarget, tf.bool))
    t_accuracy = tf.count_nonzero(same_test) / tf.constant(testTarget).shape[0]
    with tf.Session() as sess:
        test_accuracy.append(sess.run(t_accuracy))
    
    x = np.arange(len(normal_loss))
    plt.plot(x, np.array(loss), label='default_gradient')
    plt.plot(x, np.array(normal_loss), label='normal_gradient')
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()
    
    
    return (loss, normal_loss, validation_accuracy, test_accuracy)
    
def pickle_IO(ttlr):
    with open('snapshot.pkl', 'wb')  as file:
        _pickle.dump(ttlr, file, protocol=-1)
    with open('snapshot.pkl', 'rb') as file:
        ttlr_2 = _pickle.load(file)



if __name__ == '__main__':
#    plt_ttlr = tuning_the_learning_rate()
#    losses = effect_of_minibatch_size()
    (valid, test) = generalization()
    (l, nl, valid2, test2) = sgd_vs_normal_equation()
