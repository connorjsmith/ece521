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

def plot(y_hat_list, name):
    
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    
    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)
    
    # Plot mse
    y = []
    for i in range(0,len(y_hat_list)):
        y.append(tf.transpose(y_hat_list[i][0]).eval())
        plt.plot(y[i], linewidth=1.0, linestyle="-", label= r'$\eta$' +" = " + str(y_hat_list[i][1]))
    
    # Set limits and ticks
    plt.xlabel("# of epochs")
    plt.ylabel("MSE")
#    plt.xlim(0.0, 11.0)
#    plt.xticks(np.linspace(0, 11, 12, endpoint=True))
#    plt.ylim(-2.0, 10.0)
#    plt.yticks(np.linspace(-2, 10, 13, endpoint=True))
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Save figure to file
    plt.savefig(name, format="pdf")
    
    # Show result on screen
    plt.show()
        
def MSE_loss(weights, x, y, decay_coeff):
    error = tf.transpose(tf.matmul(weights, x, True)) - y
    mse_loss = ( tf.reduce_sum(tf.square(error)) / (2*tf.cast((tf.shape(y)[0]), dtype=tf.float64) ) ) + decay_coeff*tf.matmul(weights,weights,True) / 2
    return mse_loss
    
def MSE_gradient(weights, x, y, decay_coeff):
    error = tf.transpose(tf.matmul(weights, x, True)) - y
    return ( tf.transpose(error) * x ) + decay_coeff * weights
    
def linearize_data(data):
    shape = np.shape(data)
    return np.reshape(data, [shape[0], shape[1]*shape[2]])
        
def SGD(xTrain, yTrain, batchSize, iters, learning_rate, decay_coefficient):

    decay_coefficient = tf.constant(decay_coefficient, dtype=tf.float64)
    
    # Prepare x, y vector and initialize weights
    xTrain_linear = tf.transpose(tf.constant(linearize_data(xTrain)))
    x = tf.pad(xTrain_linear,[[1,0],[0,0]], "CONSTANT", constant_values = 1)
    y = tf.cast(tf.constant(yTrain, shape=[np.shape(yTrain)[0],1]), dtype=tf.float64)
    weights = tf.constant(0., shape=[np.shape(xTrain_linear)[0]+1, 1], dtype=tf.float64)
    
    
    minibatches = int(np.shape(yTrain)[0]/batchSize)

    indeces = np.arange(np.shape(yTrain)[0])
    loss_per_epoch = []
        
    for i in range(0, iters):
        current_minibatch = i % minibatches

        # Reshuffle after an epoch is completed and record loss
        if current_minibatch == 0:
            training_order = np.random.permutation(indeces)
            loss_per_epoch.append(MSE_loss(weights, x, y, decay_coefficient))
#            x_minibatch = tf.train.shuffle_batch([tf.transpose(x)], batch_size = batchSize, enqueue_many = True, capacity=np.shape(yTrain)[0], min_after_dequeue=batchSize, allow_smaller_final_batch=True)
#            y_minibatch = tf.train.shuffle_batch([tf.transpose(y)], batch_size = batchSize, enqueue_many = True, capacity=np.shape(yTrain)[0], min_after_dequeue=batchSize, allow_smaller_final_batch=True)
        
        x_minibatch = tf.gather(x, training_order[current_minibatch*batchSize : (current_minibatch+1)*batchSize], axis=tf.constant(1))
        y_minibatch = tf.gather(y, training_order[current_minibatch*batchSize : (current_minibatch+1)*batchSize])

        gradient = MSE_gradient(weights, x_minibatch, y_minibatch, decay_coefficient)
        
        weights = weights - learning_rate*tf.reduce_sum(gradient,1,True) / tf.cast(tf.shape(gradient)[1], dtype=tf.float64)

    return (weights, loss_per_epoch)


def tuning_the_learning_rate():
    # set up parameters
    iters = 200
    batch_size = 500
    learning_rates = [0.005, 0.001, 0.0001]
    decay_coefficient = 0

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    results = []
    
    for i in range(0,len(learning_rates)):
        results.append(SGD(trainData, trainTarget, batch_size, iters, learning_rates[i], decay_coefficient) + (learning_rates[i],))
        
    return results


def effect_of_minibatch_size():
    iters = 20000
    batch_sizes = [500, 1500, 3500]
    learning_rate = 0 # TODO get from tuning_the_learning_rate
    decay_coefficient = 0

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    # TODO finish me


def generalization():
    iters = 20000
    batch_size = 500
    learning_rate = 0.005
    decay_coefficients = [0.0, 0.001, 0.1, 1]

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    # TODO finish me

def sgd_vs_normal_equation():
    return 1
    # TODO finish me
    

if __name__ == '__main__':
    session = tf.InteractiveSession()
    [trainData, trainTarget, validData, validTarget, testData, testTarget] = load_data()
#    (a, b) = SGD(trainData, trainTarget, 500, 7, 1, 0)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    ttlr = tuning_the_learning_rate()
    coord.request_stop()
    coord.join(threads)