import sklearn
import numpy as np
import planar_utils
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt
from model2L import forwardProp, backwardProp, sigmoid, relup, relu, model

if __name__ == "__main__":
    X, Y = load_planar_dataset()
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]
    # YOUR CODE ENDS HERE

    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (m))

    ##Logsitic Regression

    clf = sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T, Y.T);

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")

    ##2 layer NN

    pred = model(X, Y, iterations=200, learning_rate=0.01)
    print("pred", pred.shape)
    pred = np.where(pred > 0.5, 1, 0).reshape(400)
    print('Accuracy of two layer NN: %d ' % float(
        (np.dot(Y, pred) + np.dot(1 - Y, 1 - pred)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")
