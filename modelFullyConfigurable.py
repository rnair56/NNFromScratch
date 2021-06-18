"""
n_l - number of layers
n_h - size of each layer
n_x - size of input layer

initialize the weights and biases matrix
forward propagation
    -
create the activation functions
 - backward mode - derivative of the activation fn
 -
"""
import numpy as np
import pandas as pd


def initialize_layers(n_l, n_h):
    w = {}
    b = {}
    # n_h = [10, 4, 1]
    # n_l = 3
    for i in range(1, n_l):
        w["w" + str(i)] = np.random.random((n_h[i - 1], n_h[i]))
        b["b" + str(i)] = np.zeros((n_h[i], 1))
        ##cols has the no of samples
        # rows has the number of dims
        # have number of input dims and no of neurons per layer
        # w1 = np.random((n_x, n_h[1]))
        # w2 = np.random((n_h[1], n_h[2]))
        # n_h = [10, 4, 1]
        # if n_l = 3
        # iter == 1 .. 2
        # i = i - 1 = 0
        # n_l[0] = 10; n_l[1] = 4;
        # i = 2
        # n_l[1] = 4 n_l[2] = 1
    return w, b


def relu(z):
    z[z < 0] = 0
    return z


def activationSelect(z, act="sigmoid"):
    if act == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif act == "tanh":
        return np.tanh(z)
    elif act == "relu":
        return relu(z)
    else:
        return None


def forward_propagation(w, b, X, n_l, activations):
    # z1 = np.dot(w["w" + str(1)].T, X) + b["b" + str(1)]
    # a = {"a1": activationSelect(z1, activations[0])}
    z = {}
    cache = {"a1": X}

    ## a1 done outside of loop
    ##w and b is indexed from range(1, n_l)
    for i in range(1, n_l):
        z["z" + str(i)] = np.dot(w["w" + str(i)].T, a["a" + str(i)]) + b["b" + str(i)]
        cache["a" + str(i)] = activationSelect(z["z" + str(i)], activations[i - 1])
    return cache, cache["a" + str(n_l - 1)]


def backward_propagation(y, ypred, a, w, m, activations, loss="logLossgrad"):
    dw = {}
    db = {}
    if loss == "logLossgrad":
        daL = logLossgrad(y, ypred)

    for i in reversed(range(1, n_l)):
        ##n_l was sigmoid
        dz = backprop(daL, a["a" + str(i)], activations[i - 1])
        dw["dw" + str(i)] = np.multiply(dz, a["a" + str(i - 1)])
        db["db" + str(i)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        daL = np.dot(w["w" + str(i)].T, dz)

    return dw, db


"""
for every layer what are the common calcs
- dw = dz * w
- db = 
- daprev = dzl * wl
 
"""


def backprop(daprev, a, act):
    if act == "sigmoid":
        return np.multiply(daprev, np.multiply(a, (1 - a)))
    elif act == "tanh":
        return np.multiply(daprev, (1 - np.power(a, 2)))
    elif act == "relu":
        return daprev * 1
    else:
        return None


def LogLossGrad(y, ypred):
    return -(np.divide(y, ypred) - np.divide((1 - y), (1 - ypred)))

def updates(W, b, grads, learning_rate):
    for i in range(1, n_l):
        W["w" + str(i)] = W["w" + str(i)] - np.multiply(learning_rate, grads["dw3"].T)


    #print("W3 shape", W["w3"].shape, " grads w3", grads["dw3"].shape)
    W["w2"] = W["w2"] - learning_rate * grads["dw2"].T
    W["w1"] = W["w1"] - learning_rate * grads["dw1"].T

    b["b1"] = b["b1"] - learning_rate * grads["db1"]
    b["b2"] = b["b2"] - learning_rate * grads["db2"]
    b["b3"] = b["b3"] - learning_rate * grads["db3"]
    return W, b

def modelFF(X, Y, n_h_i, n_l, activations, learning_rate=0.001, iterations=10):
    n_h = [X.shape[0]]
    n_h.append(n_h_i)

    w, b = initialize_layers(n_l=n_l, n_h=n_h)
    m = X.shape[1]

    for i in range(0, iterations):
        cache, pred = forward_propagation(w, b, X, n_l, activations)
        # loss = crossEntropyLoss(a["a3"], Y, m)
        pred = np.where(pred > 0.5, 1, 0)
        ##losssk = log_loss(Y, pred)
        dw, db = backward_propagation(Y, pred, cache, w, m, activations, loss="logLossgrad")


        w, b = updates(w, b, grads, learning_rate)
    # print(1)
    aPred = forwardProp(X, Y, w, b)
    return aPred["a3"]
