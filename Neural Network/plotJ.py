import numpy as np
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

global accuracyy
global f11
global Jarray
accuracyy = 0
f11 = 0
Jarray = []


def model(X, Y, Xtest, Ytest, layerSizes, numiter, lamb, alph, epsilon, testMode=False):
    global accuracyy, f11
    L = len(layerSizes)  # number of layers
    n = len(X[0])  # number of instances
    nTest = len(Xtest[0])
    numClass = len(Y)

    def sigmoid(x):
        return 1 / (1 + pow(math.e, -x))
        # return np.maximum(0, x)

    def prepareWeights():
        theta = {}
        for i in range(1, L):
            # theta[i] = np.random.randn(layerSizes[i], layerSizes[i - 1] + 1)  # as layerSizes start from 0
            theta[i] = np.random.normal(0, 1, size=(layerSizes[i], layerSizes[i - 1] + 1))
            # bias term to zero
            for r in range(len(theta[i])):
                theta[i][r][0] = 0
        return theta

    def forwardPropagation(theta, testData=False):
        a = {1: np.insert(X if not testData else Xtest, 0, np.ones(n if not testData else nTest), 0)}
        for k in range(2, L):
            z = np.dot(theta[k - 1], a[k - 1])
            a[k] = sigmoid(z)
            a[k] = np.insert(a[k], 0, np.ones(n if not testData else nTest), 0)
        a[L] = sigmoid(np.dot(theta[L - 1], a[L - 1]))
        return a

    def Jcost(theta, a, testData=False):
        pred = a[L]
        J = -np.multiply(Y if not testData else Ytest, np.log(pred)) - np.multiply(1 - (Y if not testData else Ytest),
                                                                                   np.log(1 - pred))
        J = np.sum(J) / (n if not testData else nTest)
        S = 0
        for l in range(1, L):
            ss = 0
            for i in range(1, len(theta[l])):
                # skip the first row of theta[l]
                ss += np.sum(np.square(theta[l][i]))
            S += ss
        S *= (lamb / (2 * (n if not testData else nTest)))
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

    def computeAcc(theta, testData=False):
        pred = forwardPropagation(theta, testData=testData)[L]
        acc = 0
        F1 = 0  # take avg later
        for trueClass in range(numClass):
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i in range(n if not testData else nTest):  # number of instances
                maxClass = 0
                for j in range(numClass):
                    if pred[j][i] > pred[maxClass][i]:
                        maxClass = j  # prediction
                if (Y if not testData else Ytest)[maxClass][i] == 1 and maxClass == trueClass:  # true prediction
                    TP += 1
                if (Y if not testData else Ytest)[maxClass][i] == 1 and maxClass != trueClass:
                    TN += 1
                if (Y if not testData else Ytest)[maxClass][i] == 0 and maxClass == trueClass:
                    FN += 1
                if (Y if not testData else Ytest)[maxClass][i] == 0 and maxClass != trueClass:
                    FP += 1
            curAcc = (TP + TN) / (TP + FP + TN + FN)
            curF1 = TP / (TP + (FP + FN) / 2)
            acc += curAcc
            F1 += curF1
        return {
            'Accuracy': acc / numClass,
            'F1': F1 / numClass
        }

    def computeJ(theta):
        pred = forwardPropagation(theta, testData=True)
        Jarray.append(Jcost(theta, pred, testData=True))

    weights = prepareWeights()
    for _ in range(numiter):
        a = forwardPropagation(weights)
        backPropagation(weights, a)
    testResult = computeAcc(weights, testData=True)
    print('Accuracy against test data: ', testResult)
    # accuracyy += testResult['Accuracy']
    # f11 += testResult['F1']
    computeJ(weights)
    return weights


def processData(name, percent=100):
    if name == 'wine':
        data = pd.read_csv('./datasets/hw3_wine.csv', sep='\t').sample(frac=1).values
        Y = data[:, 0]
        X = data[:, 1:]
        first = 13
        last = 3
        classes = [1, 2, 3]
    if name == 'cancer':
        data = pd.read_csv('./datasets/hw3_cancer.csv', sep='\t').sample(frac=1).values
        Y = data[:, 9]
        X = data[:, :9]
        first = 9
        last = 2
        classes = [0, 1]
    if name == 'vote':
        data = pd.read_csv('./datasets/hw3_house_votes_84.csv', sep=',').sample(frac=1).values
        Y = data[:, 16]
        X = data[:, :16]
        first = 16
        last = 2
        classes = [0, 1]
    if name == 'contraceptive':
        data = pd.read_csv('./datasets/cmc.csv', sep=',').sample(frac=1).values
        Y = data[:, 9]
        X = data[:, :9]
        first = 9
        last = 3
        classes = [1, 2, 3]
    numIns = int(percent * (len(X)) / 100)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X[:numIns], Y[:numIns], test_size=0.2, shuffle=True)
    # one hot coding Y
    pYtrain = np.array([np.zeros(len(Ytrain))] * len(classes))
    for i, className in enumerate(Ytrain):
        index = classes.index(className)
        pYtrain[index][i] = 1
    pYtest = np.array([np.zeros(len(Ytest))] * len(classes))
    for i, className in enumerate(Ytest):
        index = classes.index(className)
        pYtest[index][i] = 1
    return preprocessing.normalize(Xtrain), pYtrain, preprocessing.normalize(Xtest), pYtest, first, last


x = []
for i in range(5, 105, 10):
    # change contraceptive to wine, vote, or cancer
    Xtrain, Ytrain, Xtest, Ytest, first, last = processData('contraceptive',percent=i)
    lamb = 0.0001
    alph = 1.05
    epsilon = 0.00001  # stopping criteria for jcost
    numiter = 5000
    x.append(i)
    model(Xtrain.T, Ytrain, Xtest.T, Ytest, layerSizes=[first, 16, 8, last], numiter=numiter, lamb=lamb,
          alph=alph, epsilon=epsilon, testMode=False)

print(Jarray)


def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('Percent of the dataset used (%)')
    plt.ylabel('Loss value')
    plt.show()


plot(x, Jarray)
# wine
# lamb = 0.25
# alph = 2.5
# iter = 500
# layer = [13,4,3]
# lamb = 0.05
# alph = 0.01
# epsilon = 0.00001  # stopping criteria for jcost
# numiter = 4000
# layerSize = [128, 64, 32]

# # vote
# lamb = 0.4
# alph = 0.8
# iter = 300
# layer = [16, 4, 2]

# # cancer
# lamb = 0.25
# alph = 2
# iter =200
# layrer = [9,4,2]
# Y = np.array([1, 1, 1, 1, 1])
# a = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
# print(a - Y)
# data = load_boston()
# X = preprocessing.normalize(data["data"])
# Y = preprocessing.normalize(data["target"])
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# layer_sizes = [2, 4, 3, 2]
# learning_rate = 0.0001
#
# weights = model(X_train.T, Y_train.T, layerSizes=layer_sizes, lamb=0.25, alph=learning_rate)
# print(weights)
#
# X1 = np.array([[0.32, .68], [.83, .02]])
# Y1 = np.array([[.75, .98], [.75, .28]])
# a = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# def sigmoid(x):
#     return 1 / (1 + pow(math.e, -x))
# b = sigmoid(a)
# print(b)
# sum = np.sum(a.T, axis=1)
# print(sum)
# a = np.delete(a, 0, 0)
# print(a)
# b = np.insert(a,0, [0,0],0)
# print(b)