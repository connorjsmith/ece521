import tensorflow as tf
import numpy as np
from choosing_nearest_neighbours import *
import matplotlib.pyplot as plt


def get_dataset():
    np.random.seed(521)
    Data = np.linspace(1.0, 10.0, num=100) [:, np.newaxis]
    Target = np.sin(Data) + 0.1*np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    
    return Data, Target, randIdx

def kNN_regression(test_points, in_features, targets, k):
    r_star = get_responsibility_matrix(in_features, test_points, k)
    targets = tf.constant(targets, tf.float64)
    return tf.matmul(targets, r_star, True, True)  
    
def calculate_mse(prediction, targets):
    size = targets.shape[0]
    return tf.reduce_sum(tf.square(targets-prediction)) / (2*size)
    
def prepare_data():
    
    Data, Target, randIdx = get_dataset()
    
    trainData = Data[randIdx[:80]]
    trainTarget = Target[randIdx[:80]]
    
    validData = Data[randIdx[80:90]]
    validTarget = Target[randIdx[80:90]]
    
    testData = Data[randIdx[90:100]]
    testTarget = Target[randIdx[90:100]]
    
    return [(trainData, trainTarget, "Training"), (validData, validTarget, "Validation"), (testData, testTarget, "Test")]      

def regression():
    
    k_list = [1,3,5,50]
    in_out_pairs = prepare_data()
    y_hat = []
    test_points = np.linspace(0.0,11.0,num = 1000)[:,np.newaxis]

    for k in k_list:
        prediction = kNN_regression(test_points, in_out_pairs[0][0], in_out_pairs[0][1], k)
        y_hat.append((prediction, k))
        
    return y_hat

def plot_combined(y_hat):
    
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    
    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)
    
    (data, targets, __) = get_dataset();
    x = np.linspace(0.0,11.0,num = 1000)[:,np.newaxis]
    y1 = tf.transpose(y_hat[0][0]).eval()
    y2 = tf.transpose(y_hat[1][0]).eval()
    y3 = tf.transpose(y_hat[2][0]).eval()
    y4 = tf.transpose(y_hat[3][0]).eval()
    
    plt.plot(x, y1, color="blue", linewidth=1.0, linestyle="-", label="k = " + str(y_hat[0][1]))    
    plt.plot(x, y2, color="green", linewidth=1.0, linestyle="-", label="k = " + str(y_hat[1][1]))   
    plt.plot(x, y3, color="red", linewidth=1.0, linestyle="-", label="k = " + str(y_hat[2][1]))
    plt.plot(x, y4, color="purple", linewidth=1.0, linestyle="-", label="k = " + str(y_hat[3][1]))
    
    plt.plot(data, targets, 'o', markersize=4., label="Dataset")
    
    # Set x limits
    plt.xlim(0.0, 11.0)
    
    # Set x ticks
    plt.xticks(np.linspace(0, 11, 12, endpoint=True))
    
    # Set y limits
    plt.ylim(-2.0, 10.0)
    
    # Set y ticks
    plt.yticks(np.linspace(-2, 10, 13, endpoint=True))
    
    # Legend
    plt.legend(loc='upper left')
    
    # Save figure using 72 dots per inch
    plt.savefig("combined.pdf", format="pdf")
    
    # Show result on screen
    plt.show()
    
def plot_individual(y_hat,k_ind):
    
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    
    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)
    
    (data, targets, __) = get_dataset();
    x = np.linspace(0.0,11.0,num = 1000)[:,np.newaxis]
    y = tf.transpose(y_hat[k_ind][0]).eval()
    k = y_hat[k_ind][1]
    
    plt.plot(x, y, color="red", linewidth=1.0, linestyle="-", label="k = " + str(k))    
    
    plt.plot(data, targets, 'o', markersize=4., label="Dataset")
    
    # Set x limits
    plt.xlim(0.0, 11.0)
    
    # Set x ticks
    plt.xticks(np.linspace(0, 11, 12, endpoint=True))
    
    # Set y limits
    plt.ylim(-2.0, 10.0)
    
    # Set y ticks
    plt.yticks(np.linspace(-2, 10, 13, endpoint=True))
    
    # Legend
    plt.legend(loc='upper left')
    
    # Save figure using 72 dots per inch
    plt.savefig("k" + str(k) + "-plot.pdf", format="pdf")
    
    # Show result on screen
    plt.show() 
    
if __name__ == '__main__':
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    test = regression()
