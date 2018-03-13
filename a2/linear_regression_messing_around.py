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
#    y = []
#    for i in range(0,len(y_hat_list)):
#        y.append(tf.transpose(y_hat_list[i][0]).eval())
#        plt.plot(y[i], linewidth=1.0, linestyle="-", label= r'$\eta$' +" = " + str(y_hat_list[i][1]))
    
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
#    print(weights)    
#    print(error)
#    print(x)
#    print(( tf.transpose(error) * x ))
#    print(decay_coeff * weights)
#    print((( tf.transpose(error) * x ) + decay_coeff * weights))
    return ( tf.transpose(error) * x ) + decay_coeff * weights
    
def linearize_data(data):
    shape = np.shape(data)
    reshaped_data = np.reshape(data, [shape[0], shape[1]*shape[2]])
    return np.pad(np.transpose(reshaped_data), ((1,0),(0,0)), 'constant', constant_values = 1 )

def SGD(data_shape, batchSize, iters):

    # Prepare x, y vector and initialize weights
#    xTrain_linear = tf.transpose(tf.constant(linearize_data(xTrain)))
#    x = tf.pad(xTrain_linear,[[1,0],[0,0]], "CONSTANT", constant_values = 1)
#    y = tf.cast(tf.constant(yTrain, shape=[np.shape(yTrain)[0],1]), dtype=tf.float64)
    x_SGD = tf.placeholder(tf.float64, data_shape)
    y_SGD = tf.placeholder(tf.float64, [data_shape[1], 1])
    learning_rate = tf.placeholder(tf.float64)
    decay_coefficient = tf.placeholder(tf.float64)
#    weights = tf.Variable(tf.zeros([data_shape[0], 1], tf.float64))
    weights = tf.zeros([data_shape[0], 1], tf.float64)
    
    minibatches = int(data_shape[1]/batchSize)
    loss_per_epoch = []
        
    for i in range(0, iters):
        current_minibatch = i % minibatches

        # Reshuffle after an epoch is completed and record loss
        if current_minibatch == 0:
            x_shuffled = tf.transpose(tf.random_shuffle(tf.transpose(x_SGD)))
            y_shuffled = tf.transpose(tf.random_shuffle(tf.transpose(y_SGD)))
            loss_per_epoch.append(MSE_loss(weights, x_SGD, y_SGD, decay_coefficient))
#        loss_per_epoch.append(weights)
        x_minibatch = x_shuffled[:, current_minibatch*batchSize : (current_minibatch+1)*batchSize]
        y_minibatch = y_shuffled[current_minibatch*batchSize : (current_minibatch+1)*batchSize]
        
        gradient = MSE_gradient(weights, x_minibatch, y_minibatch, decay_coefficient)
        
        weights = weights - learning_rate*tf.reduce_sum(gradient,1,True) / tf.cast(tf.shape(gradient)[1], dtype=tf.float64)

    return (x_SGD, y_SGD, batchSize, learning_rate, decay_coefficient, weights, loss_per_epoch)

def SGD_epoch2(data_shape, batchSize, iters):

    # Prepare x, y vector and initialize weights
#    xTrain_linear = tf.transpose(tf.constant(linearize_data(xTrain)))
#    x = tf.pad(xTrain_linear,[[1,0],[0,0]], "CONSTANT", constant_values = 1)
#    y = tf.cast(tf.constant(yTrain, shape=[np.shape(yTrain)[0],1]), dtype=tf.float64)
    x_SGD = tf.placeholder(tf.float64, data_shape)
    y_SGD = tf.placeholder(tf.float64, [data_shape[1], 1])
    learning_rate = tf.placeholder(tf.float64)
    decay_coefficient = tf.placeholder(tf.float64)
    weights = tf.Variable(tf.zeros([data_shape[0], 1], tf.float64))
    
    minibatches = int(data_shape[1]/batchSize)
    w_ops = []
        
    for i in range(0, iters):
        current_minibatch = i % minibatches

        # Reshuffle after an epoch is completed and record loss
        if current_minibatch == 0:
            x_shuffled = tf.transpose(tf.random_shuffle(tf.transpose(x_SGD)))
            y_shuffled = tf.transpose(tf.random_shuffle(tf.transpose(y_SGD)))

        x_minibatch = x_shuffled[:, current_minibatch*batchSize : (current_minibatch+1)*batchSize]
        y_minibatch = y_shuffled[current_minibatch*batchSize : (current_minibatch+1)*batchSize]
        
        gradient = MSE_gradient(weights, x_minibatch, y_minibatch, decay_coefficient)
        
        w_ops.append(weights.assign_sub(learning_rate*tf.reduce_sum(gradient,1,True) / tf.cast(tf.shape(gradient)[1], dtype=tf.float64)))

    return (x_SGD, y_SGD, learning_rate, decay_coefficient, w_ops)


def SGD_epoch(data_shape, batchSize, iters):

    # Prepare x, y vector and initialize weights
    x_SGD = tf.placeholder(tf.float64, data_shape)
    y_SGD = tf.placeholder(tf.float64, [data_shape[1], 1])
    learning_rate = tf.placeholder(tf.float64)
    decay_coefficient = tf.placeholder(tf.float64)
    weights = tf.Variable(tf.zeros([data_shape[0], 1], tf.float64))
    
    x_shuffled = tf.transpose(tf.random_shuffle(tf.transpose(x_SGD)))
    y_shuffled = tf.transpose(tf.random_shuffle(tf.transpose(y_SGD)))

    for i in range(0, iters):

        x_minibatch = x_shuffled[:, iters*batchSize : (iters+1)*batchSize]
        y_minibatch = y_shuffled[iters*batchSize : (iters+1)*batchSize]
        
        gradient = MSE_gradient(weights, x_minibatch, y_minibatch, decay_coefficient)
        
        weights.assign_sub(learning_rate*tf.reduce_sum(gradient,1,True) / tf.cast(tf.shape(gradient)[1], dtype=tf.float64))

    return (x_SGD, y_SGD, learning_rate, decay_coefficient, weights, loss_per_epoch)

    
def tuning_the_learning_rate():
    # set up parameters
    iters = 400
    batch_size = 500
    learning_rates = [0.005, 0.001, 0.0001]
    decay_coefficient = 0

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    results = []
    
    with tf.Session() as sess:
         [trainData, trainTarget, validData, validTarget, testData, testTarget] = load_data()
         x_linearized = linearize_data(trainData)
         [x_SGD, y_SGD, learning_rate, decay_coefficient, w_ops] = SGD_epoch2(np.shape(x_linearized), 500, 8)
         weights = tf.Variable(tf.zeros([np.shape(x_linearized)[0], 1], tf.float64))
         sess.run(tf.global_variables_initializer())
         sess.run(tf.local_variables_initializer())
         
         for i in range(0,len(w_ops)):
             print(i % 7)
             if i % 7 == 0:
                 results.append(MSE_loss(weights, x_linearized, trainTarget, 0))
        
             weights = sess.run(w_ops[i], feed_dict={x_SGD: x_linearized, y_SGD: trainTarget, learning_rate: 0.005, decay_coefficient: 0})       

#        results.append(SGD(trainData, trainTarget, batch_size, iters, learning_rates[i], decay_coefficient) + (learning_rates[i],))
        
    return (results, a)


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
#    sess = tf.InteractiveSession()
#    [trainData, trainTarget, validData, validTarget, testData, testTarget] = load_data()
##    (a, b) = SGD(trainData, trainTarget, 500, 7, 1, 0)
     [a, b] = tuning_the_learning_rate()
#    x_linearized = linearize_data(trainData)
#    [x_SGD, y_SGD, batchSize, learning_rate, decay_coefficient, w, lpe] = SGD(np.shape(x_linearized), 500, 8)
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
#    [a,b] = sess.run([w, lpe], feed_dict={x_SGD: x_linearized, y_SGD: trainTarget, learning_rate: 0.005, decay_coefficient: 0})