def load_data()
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


def linear_regression(xTrain, yTrain, batchSize, iters):
    epochs = len(xTrain)/batchSize


def tuning_the_learning_rate():
    # set up parameters
    iters = 20000
    batch_size = 500
    learning_rates = [0.005, 0.001, 0.0001]
    decay_coefficient = 0

    (trainData, trainTarget, 
     testData, testTarget, 
     validData, validTarget) = load_data()
    # TODO finish me


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
    # TODO finish me
