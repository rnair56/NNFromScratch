import numpy as np
import pandas as pd
import math


def relu(x) -> object:
    if x > 0:
        return x
    else:
        return 0.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relup(x):
    rows = x.shape[0]
    cols = x.shape[1]
    xx = pd.DataFrame()
    for i in range(0, rows):
        for j in range(0, cols):
            xx.loc[i, j] = relu(x[i, j])
    return xx


# def relunp(x):
#    return relu(np.array(x))


def initialize_weights(X, l=2):
    w = {}
    b = {}
    m = X.shape
    w1 = np.random.random((m[0], 64))  ## #input cols, neurons
    # print(w1.shape)
    w2 = np.random.random((64, 16))
    w3 = np.random.random((16, 1))

    # b1 = np.random.random((64, 1))
    # b2 = np.random.random((16, 1))
    # b3 = np.random.random((1, 1))
    b1 = np.zeros((64, 1))
    b2 = np.zeros((16, 1))
    b3 = np.zeros((1, 1))
    w["w1"] = w1
    w["w2"] = w2
    w["w3"] = w3
    b["b1"] = b1
    b["b2"] = b2
    b["b3"] = b3
    return w, b


def forwardProp(X, Y, w, b):
    """
    X
    Y
    l  - number of hidden layers hardcoded
    :return:
    for 2 layers - input shape w . 64, 1000
    shape of first layer will have to be in accordance to the input size
    """
    # m = X.shape
    a = {}
    ##w1 = np.random.random((m[0], 1))  ## 64,1

    z1 = np.dot(w["w1"].T, X) + b["b1"]  ##for sample size 1000, 1,1000
    #print("z1", z1.shape)
    # a1 = relup(z1)
    a1 = np.tanh(z1)
    #print("a1", a1.shape)

    z2 = np.dot(w["w2"].T, a1) + b["b2"]
    a2 = np.tanh(z2)

    z3 = np.dot(w["w3"].T, a2) + b["b3"]
    #print("z3.shape", z3.shape)
    #print("w3.shape", w["w3"].shape)
    #print("a2.shape", a2.shape)

    a3 = sigmoid(z3)
    a["a1"] = a1
    a["a2"] = a2
    a["a3"] = a3

    return a


def backwardProp(a, Y, x, w, m):
    """

    :return:
    """
    grads = {}
    dz3 = Y - a["a3"]
    dw3 = (1 / m) * np.dot(dz3, a["a2"].T)
    grads["dw3"] = dw3
    #print("dw3.shape = ", dw3.shape)

    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)
    grads["db3"] = db3

    ##dz2 = np.where(a["a2"] > 0, 1, 0)
    ##dw2 = dz2 * a["a1"]
    ##grads["dw2"] = dw2
    dz2 = np.multiply(np.dot(w["w3"], dz3), np.power(a["a2"], 2))
    dw2 = (1 / m) * np.dot(dz2, a["a1"].T)
    #print("dw2.shape = ", dw2.shape)
    grads["dw2"] = dw2

    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    grads["db2"] = db2

    # dz1 = np.where(a["a1"] > 0, 1, 0)
    # dw1 = dz1 * x
    # grads["dw1"] = dw1
    dz1 = np.multiply(np.dot(w["w2"], dz2), np.power(a["a1"], 2))
    dw1 = (1 / m) * np.dot(dz1, x.T)
    grads["dw1"] = dw1

    #print("dw1 shape", dw1.shape)
    #print("dz1 shape", dz1.shape)

    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    grads["db1"] = db1

    return grads


def updates(W, b, grads, learning_rate):

    W["w3"] = W["w3"] - np.multiply(learning_rate, grads["dw3"].T)
    W["w2"] = W["w2"] - learning_rate * grads["dw2"].T
    W["w1"] = W["w1"] - learning_rate * grads["dw1"].T

    b["b1"] = b["b1"] - learning_rate * grads["db1"]
    b["b2"] = b["b2"] - learning_rate * grads["db2"]
    b["b3"] = b["b3"] - learning_rate * grads["db3"]
    return W, b


def crossEntropyLoss(a, y, m):
    loss = np.multiply(y, np.log(a)) + np.multiply((1 - y), np.log(1 - a))
    #print(loss.shape)
    # loss = np.where(loss == np.NAN, 0, loss)
    # loss = [0 if i == np.NAN else i for i in loss]
    # loss = [0 if np.isinf(-i) else i for i in loss]
    #[print(len(i)) for i in loss]
    loss = -(1 / m) * np.sum(loss)
    return loss


def model(X, Y, learning_rate=0.001, iterations=10):
    w, b = initialize_weights(X)
    m = X.shape[1]
    for i in range(0, iterations):

        a = forwardProp(X, Y, w, b)
        loss = crossEntropyLoss(a["a3"], Y, m)
        grads = backwardProp(a, Y, X, w, m)
        print("current loss is ", loss)

        w, b = updates(w, b, grads, learning_rate)
    #print(1)
    aPred = forwardProp(X, Y, w, b)
    return aPred["a3"]
