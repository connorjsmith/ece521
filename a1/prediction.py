import tensorflow as tf
import numpy as np
from choosing_nearest_neighbours import *


def get_dataset():
  np.random.seed(521)
  Data = np.linspace(1.0, 10.0, num=100) [:, np.newaxis]
  Target = np.sin(Data) + 0.1*np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
  randIdx = np.arange(100)
  np.random.shuffle(randIdx)

  return Data, Target, randIdx

def prediction():
  k_list = [1,3,5,50]
  Data, Target, randIdx = get_dataset()

  trainData = Data[randIdx[:80]]
  trainTarget = Target[randIdx[:80]]

  validData = Data[randIdx[80:90]]
  validTarget = Target[randIdx[80:90]]

  testData = Data[randIdx[90:100]]
  testTarget = Target[randIdx[90:100]]

  in_out_pairs = [(trainData, trainTarget, "train"), (validData, validTarget, "valid"), (testData, testTarget, "test")]


  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for k in k_list:
      print("k=%d" % k)
      for X, Y, name in in_out_pairs:
        r_matrix = get_responsibility_matrix(trainData, X, k)
        y_preds = tf.reduce_sum(tf.transpose(trainTarget) * r_matrix, axis=-1)
        y_preds = tf.reshape(y_preds, [-1,1])
        error = tf.reduce_sum(tf.square(Y-y_preds)) / (2*X.shape[0])
        print("    set=%s error=%lf" % (name,error.eval()))
      print("\n")
    
        

if __name__ == '__main__':
    prediction()
