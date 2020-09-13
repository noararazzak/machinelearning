import numpy as np
from sklearn.datasets import make_friedman1
from matplotlib import pyplot

#This is Linear  Regression with 1 or 2 hidden layer that uses
#both RELU, Sigmoid, Tanh and Leaky RELU as activation functions. Loss function
#is a simple mean sqaured error function

#Generating and plotting the data

def make_data_weights_biases(neurons, twolayers):
    X, y = make_friedman1(n_samples=1000, n_features=5, noise=0.0, random_state=None)

    W_0= np.random.rand(X.shape[1], neurons)
    b_0 = np.zeros((1, neurons))
    if twolayers:
        W_1 = np.random.rand(neurons, neurons)
        W_2 = np.random.rand(neurons, 1)
        b_1 = np.zeros((1, neurons))
        b_2 = np.zeros((1, 1))
    else:
        W_1 = np.random.rand(neurons, 1)
        b_1 = np.zeros((1, 1))
        W_2 = None
        b_2 = None

    print("X rows: " + repr(X.shape[0]) + ", " + "X columns: " + repr(X.shape[1]))
    print("Y rows: " + repr(y.shape))
    print("W_0: " + repr (W_0) + ", " + "W_1: " + repr (W_1) + "b_0: " + repr (b_0) + "b_1: " + repr (b_1) )
    if twolayers:
        print("W_0: " + repr(W_0) + ", " + "W_1: " + repr(W_1) + "W_2: " + repr(W_2) +
              "b_0: " + repr(b_0) + "b_1: " + repr(b_1) + "b_2: " + repr(b_2))
        return X, y, W_0, W_1, W_2, b_0, b_1, b_2
    else:
        print("W_0: " + repr(W_0) + ", " + "W_1: " + repr(W_1) + "b_0: " + repr(b_0) + "b_1: " + repr(b_1))
        return X, y, W_0, W_1, b_0, b_1

