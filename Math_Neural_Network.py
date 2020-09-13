import numpy as np

#Definitions
#z1, z2, z3 are linear combinations
#W_0 (first weight)
#W_1 (second weight)
#W_2 (third weight, for 2 layers only)
#a_1 (first activation function)
#a_2 (second activation function)
#a_3 (third activation function, for 2 layers only)
#da_1 (first activation derivative function wrt z1)
#da_2 (second activation derivative function wrt z2)
#da_3 (third activation derivative function, for 2 layers only wrt z3)


#Activation functions and their derivatives
def sigmoid(x):
    denominator = 1 + np.exp(-x)
    return 1./denominator


def sigmoid_derivative(x):
    return x*(1-x)


def tanh(x):
    nominator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return nominator/denominator


def tanh_derivative(x):
    return 1. - (x*x)


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0

def leaky_relu(x, a):
    if x > 0:
        return x
    else:
        return a*x

def leaky_relu_derivative(x, a):
    if x > 0:
        return x
    else:
        return a

def softmax(x):
    exps = np.exp(x - np.max(x, axis =1, keepdims=True))
    return  exps/np.sum(exps, axis=1, keepdims=True)


# Loss Functions with one hidden layer
def mean_squared_loss_derivative_linear_function(y_hat, y):
    return y_hat - y

# Back propagation with one hidden layer
def dL_dw_2_mean_squared_loss(y, a_2, da_2, a_1, n):
    a = (2/n)*(a_2 - y)
    b = da_2
    c = a_1
    result = a*b*c
    return result

def dL_dw_1_mean_squared_loss(y, a_2, da_2, da_1, W_2, X, n):
    a = (2/n)*(a_2 - y)
    b = da_2
    c = W_2
    d = da_1
    e = X
    result = a*b*c*d*e
    return result



# Functions from mister
def tryParseINT(string):
    try:
        return int(string)
    except ValueError:
        return None


