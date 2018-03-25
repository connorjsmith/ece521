def load_notMNIST():
    with np.load("./notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]

        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        return (trainData.reshape(trainData.shape[0], -1), trainTarget,
                validData.reshape(validData.shape[0], -1), validTarget,
                testData.reshape(testData.shape[0], -1), testTarget)
