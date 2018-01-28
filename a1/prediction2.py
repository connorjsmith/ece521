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

def plot_combined(y_hat_list):
    
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    
    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)
    
    # Prepare data
    (data, targets, __) = get_dataset();
    x = np.linspace(0.0,11.0,num = 1000)[:,np.newaxis]
    
    # Plot data points
    plt.plot(data, targets, 'o', color='#7f7f7f', markersize=4., label="Dataset")
    
    # Plot regression lines
    y = []
    for i in range(0,len(y_hat_list)):
        y.append(tf.transpose(y_hat_list[i][0]).eval())
        plt.plot(x, y[i], linewidth=1.0, linestyle="-", label="k = " + str(y_hat_list[i][1]))
    
    # Set limits and ticks
    plt.xlim(0.0, 11.0)
    plt.xticks(np.linspace(0, 11, 12, endpoint=True))
    plt.ylim(-2.0, 10.0)
    plt.yticks(np.linspace(-2, 10, 13, endpoint=True))
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Save figure to file
    plt.savefig("combined.pdf", format="pdf")
    
    # Show result on screen
    plt.show()
    
def plot_individual(y_hat):

    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)
    
    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)
    
    # Prepare data
    (data, targets, __) = get_dataset();
    x = np.linspace(0.0,11.0,num = 1000)[:,np.newaxis]

    # Plot data points
    plt.plot(data, targets, 'o', color='#7f7f7f', markersize=4., label="Dataset")
    
    # Plot regression line
    y = tf.transpose(y_hat[0]).eval()
    k = y_hat[1]    
    plt.plot(x, y, color="red", linewidth=1.0, linestyle="-", label="k = " + str(k))    
    
    # Set limits and ticks
    plt.xlim(0.0, 11.0)
    plt.xticks(np.linspace(0, 11, 12, endpoint=True))
    plt.ylim(-2.0, 10.0)
    plt.yticks(np.linspace(-2, 10, 13, endpoint=True))
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Save figure to file
    plt.savefig("k" + str(k) + "-plot.pdf", format="pdf")
    
    # Show result on screen
    plt.show() 
    
if __name__ == '__main__':
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    y_hat_list = regression()
    plot_combined(y_hat_list)
    for y_hat in y_hat_list:
        plot_individual(y_hat)
