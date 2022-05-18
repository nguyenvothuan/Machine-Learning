import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd


def test(lamb, alph, X, Y, theta, layerSizes):
    n = len(X[0])
    L = len(layerSizes)
    def sigmoid(x):
        return 1 / (1 + pow(math.e, -x))
        # return np.maximum(0, x)

    def forwardPropagation():
        a = {1: np.insert(X, 0, np.ones(2), 0)}
        for k in range(2, L):
            z = np.dot(theta[k - 1], a[k - 1])
            a[k] = sigmoid(z)
            a[k] = np.insert(a[k], 0, np.ones(2), 0)
        a[L] = sigmoid(np.dot(theta[L - 1], a[L - 1]))
        print('Activation of the last layer: ')
        print(a[L])
        print('---------------')
        return a

    def Jcost(theta, a):
        pred = a[L]
        J = -np.multiply(Y, np.log(pred)) - np.multiply(1 - Y, np.log(1 - pred))
        J = np.sum(J) / 2
        S = 0
        for l in range(1, L):
            ss = 0
            for i in range(1, len(theta[l])):
                # skip the first row of theta[l]
                ss += np.sum(np.square(theta[l][i]))
            S += ss
        S *= (lamb / 4)
        print('Jcost: ', J + S)
        print('---------------')
        return J + S

    def backPropagation(theta, a):
        pred = a[L]
        delta = {L: pred - Y}
        grad = {}
        for k in reversed(range(2, L)):
            # remove first bias row
            delta[k] = np.delete(np.multiply(np.dot(theta[k].T, delta[k + 1]), np.multiply(a[k], 1 - a[k])), 0, 0)
        for k in reversed(range(1, L)):
            grad[k] = np.dot(delta[k + 1], a[k].T)
        for k in reversed(range(1, L)):
            p = lamb * theta[k]
            for i in range(len(p)):
                p[i][0] = 0
            grad[k] = (p + grad[k]) / n
        for k in reversed(range(1, L)):
            theta[k] -= alph * grad[k]
        print('Delta: ')
        print(delta)
        print('---------------')
        print('Gradient: ')
        print(grad)
        print('---------------')

    a = forwardPropagation()
    Jcost(theta, a)
    backPropagation(theta, a)


mockTheta1 = {
    1: np.array([[.4, .1], [.3, .2]]),
    2: np.array([[.7, .5, .6]])
}
mockX1 = np.array([[.13, .42]])
mockY1 = np.array([[.9, .23]])
lamb1 = 0
layerSizes1 = [1, 2, 1]

mockTheta2 = {
    1: np.array([[0.42, 0.15, .4], [.72, .1, .54], [.01, .19, .42], [.3, .35, .68]]),
    2: np.array([[.21, .67, .14, .96, .87], [.87, .42, .2, .32, .89], [.03, .56, .8, .69, .09]]),
    3: np.array([[.04, .87, .42, .53], [.17, .1, .95, .69]])
}
lamb2 = 0.25
layerSizes2 = [2, 4, 3, 2]
mockX2 = np.array([[0.32, .83], [.68, .02]])
mockY2 = np.array([[.75, .75], [.98, .28]])

test(lamb1, alph=0.0001, X=mockX1, Y=mockY1, theta=mockTheta1, layerSizes=layerSizes1)
# test(lamb2, alph=0.0001, X=mockX2, Y=mockY2, theta=mockTheta2, layerSizes=layerSizes2)
