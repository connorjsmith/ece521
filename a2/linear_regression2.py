import _pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    mse_loss = tf.reduce_mean(tf.sqrt(tf.square(Y-z)))/2
    total_loss = mse_loss + dc*tf.sqrt(tf.reduce_sum(tf.square(weights)))/2
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
                this_loss, _ = sess.run([total_loss, grads],
                                        feed_dict={X: this_batch_X, Y:this_batch_Y.astype(np.float64),
                                                   dc:np.array([decay_coefficient], dtype=np.float64)})
                this_epoch_loss += this_loss*(u-l)
                assert len(grads) == 1
                sess.run(updates,
                         feed_dict={X: this_batch_X, Y:this_batch_Y.astype(np.float64),
                                    dc:np.array([decay_coefficient], dtype=np.float64)})
            if curr_samples == 0:
                continue
            this_epoch_loss /= curr_samples
            losses.append(this_epoch_loss)
    return losses


def tuning_the_learning_rate():

    iters = 20000
    batch_size = 500
    learning_rates = [0.005, 0.001, 0.0001]
    decay_coefficient = 0.

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()

    for i in range(0,len(learning_rates)):
        loss = SGD(trainData, trainTarget, batch_size, iters, learning_rates[i], decay_coefficient)
        x = np.arange(len(loss))
        plt.plot(x, np.array(loss),label = 'learning_rate = %f'%learning_rates[i])

    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()


def effect_of_minibatch_size():
    iters = 20000
    batch_sizes = [500, 1500, 3500]
    learning_rate = 0.001 # TODO get from tuning_the_learning_rate
    decay_coefficient = 0

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    # TODO finish me
    for i in range(0,len(batch_sizes)):
        loss = SGD(trainData, trainTarget, batch_sizes[i], iters, learning_rate, decay_coefficient)
        x = np.arange(len(loss))
        plt.plot(x, np.array(loss), label='batch_size = %f' % batch_sizes[i])

    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()
    return loss


def generalization():
    iters = 20000
    batch_size = 500
    learning_rate = 0.005
    decay_coefficients = [0.0, 0.001, 0.1, 1]

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    # TODO finish me

    for i in range(0,len(decay_coefficients)):
        loss = SGD(trainData, trainTarget, batch_size, iters, learning_rate, decay_coefficients[i])
        x = np.arange(len(loss))
        plt.plot(x, np.array(loss), label='decay_coefficient = %f' % decay_coefficients[i])

    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()


def sgd_vs_normal_equation():
    iters = 20000
    batch_size = 500
    learning_rate = 0.005
    decay_coefficient = 0.001

    (trainData, trainTarget,
     testData, testTarget,
     validData, validTarget) = load_data()
    # TODO finish me

    loss = SGD(trainData, trainTarget, batch_size, iters, learning_rate, decay_coefficient)
    normal_loss = SGD(trainData, trainTarget, batch_size, iters, learning_rate, decay_coefficient, True)
    x = np.arange(len(normal_loss))
    plt.plot(x, np.array(loss), label='defaul_gradient')
    plt.plot(x, np.array(normal_loss), label='normal_gradient')
    plt.xlabel('Number of Epoch')
    plt.ylabel('MSE')
    plt.title("MSE vs Epoch")
    plt.legend()
    plt.show()
    # TODO finish me
    
def pickle_IO(ttlr):
    with open('snapshot.pkl', 'wb')  as file:
        _pickle.dump(ttlr, file, protocol=-1)
    with open('snapshot.pkl', 'rb') as file:
        ttlr_2 = _pickle.load(file)



if __name__ == '__main__':
    tuning_the_learning_rate()
    #effect_of_minibatch_size()
    #generalization()
    #sgd_vs_normal_equation()
