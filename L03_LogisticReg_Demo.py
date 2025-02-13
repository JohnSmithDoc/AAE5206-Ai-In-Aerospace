
## AAE5206, Bing WANG

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def get_data():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()

    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((X,intercept), axis=1)
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def training(X, y):
    # weights initialization
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    for i in range(100000):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta =theta - learning_rate * gradient

        ls = loss(h, y)
        print(f'loss: {ls} \t')
    return theta

def testing(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)

    h = h.round()
    return h

def demo():
    X, y = get_data()
    theta = training(X, y)

    preds = testing(X, theta)
    accuracy = (preds == y).mean()
    print('training accuracy:', accuracy)

################################
if __name__ == "__main__":
    demo()