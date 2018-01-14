import numpy as np

def euclidean_distance(X, Z):
    D = X.shape[1]
    N1 = X.shape[0]
    N2 = Z.shape[0]
    X_int = X.reshape(N1,1,D)
    Z_int = Z.reshape(1,N2,D)

    distance_pairs = X_int - Z_int # N1xN2xD matrix
    squared_euclidean_distances = np.square(distance_pairs).sum(axis=2) #  N1xN2 matrix of squared pairwise distances
    euclidean_distances = np.sqrt(squared_euclidean_distances) # element-wise squareroot

    return euclidean_distances


