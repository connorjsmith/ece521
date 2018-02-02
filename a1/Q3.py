import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from kNN_classification import *

name_task = 0
gender_task = 1

def data_segmentation(data_path, target_path, task):
# task = 0 >> select the name ID targets for face recognition task
# task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    
    target = np.load(target_path)
    
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Takes linearized picture data and puts it back into matrix form
def form_picture(data, index):
    pictures = np.reshape(data, [-1,32,32])
    return pictures[index]

def print_pictures(dataset, indeces, print_type):
    
    types = ["NN-Name-", "NN-Gender-", "OO-Name-", "OO-Gender-"]
    
    for i in range(0, len(indeces)):
        pic = np.reshape(dataset[indeces[i]], [-32,32])
        plt.imshow(pic, cmap="gray")
        
        # Save figure to file
        plt.savefig(types[print_type] + str(i) + ".pdf", format="pdf")
        
        # Show result on screen
        plt.show() 

def perform_classification(NN_type):
    
    # List of nearest neighbours
    k = [1,5,10,25,50,100,200]

    # Load dataset
    (trainData, validData, testData, trainTarget, validTarget, testTarget) = data_segmentation("data.npy", "target.npy", NN_type)
    
    # Classification based on training data/targets
    classification_training = []
    performance_training = []
    for i in range(0,len(k)):
        classifications = kNN_classification(trainData, trainData, trainTarget, k[i])
        classification_training.append(classifications)
        
        performance = classification_performance(classifications[0], trainTarget)
        performance_training.append(performance)
    
    # Classification based on validation data/targets
    classification_validation = []
    performance_validation = []
    for i in range(0,len(k)):
        classifications = kNN_classification(validData, trainData, trainTarget, k[i])
        classification_validation.append(classifications)
        
        performance = classification_performance(classifications[0], validTarget)
        performance_validation.append(performance)
        
    # Classification based on test data/targets
    classification_test = []
    performance_test = []
    for i in range(0,len(k)):
        classifications = kNN_classification(testData, trainData, trainTarget, k[i])
        classification_test.append(classifications)
        
        performance = classification_performance(classifications[0], testTarget)
        performance_test.append(performance)
    
    return (classification_training, performance_training, classification_validation, \
            performance_validation, classification_test, performance_test)
    
if __name__ == '__main__':   
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Put the task type here
    NN_type = name_task
    
    (trainData, validData, testData, trainTarget, validTarget, testTarget) = data_segmentation("data.npy", "target.npy", NN_type)
    (classification_training, performance_training, classification_validation, \
            performance_validation, classification_test, performance_test) = perform_classification(NN_type)
    
    # Picture at index = 0 is known to misclassify the name (0 instead of 3)
    if NN_type == name_task:
        print_pictures(validData, [0], NN_type+2)
        print_pictures(trainData, classification_validation[2][1][0].eval(), NN_type)
     # Picture at index = 1 is known to misclassify the gender (1 instead of 0)
    else:
        print_pictures(validData, [1], NN_type+2)
        print_pictures(trainData, classification_validation[2][1][1].eval(), NN_type)