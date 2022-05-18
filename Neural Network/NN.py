import numpy as np
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import pandas as pd

global accuracyy
global f11
accuracyy = 0
f11 = 0


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
        J = -np.multiply(Y if not testData else Ytest, np.log(pred)) - np.multiply(1 - Y if not testData else Ytest,
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

    weights = prepareWeights()
    for _ in range(numiter):
        a = forwardPropagation(weights)
        backPropagation(weights, a)
        # print('Cost: ', Jcost(weights, a), '[==========>] Accuracy: ', computeAcc(weights)['Accuracy'])
    testResult = computeAcc(weights, testData=True)
    print('Accuracy against test data: ', testResult)
    accuracyy += testResult['Accuracy']
    f11 += testResult['F1']
    return weights


def processData(name, numFolds=10):
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

    folder = StratifiedKFold(n_splits=numFolds, shuffle=True, random_state=1)
    foldIndex = folder.split(X, Y)
    kFolds = []
    for trainIndex, testIndex in foldIndex:
        Xtrain, Xtest = preprocessing.normalize(X[trainIndex]), preprocessing.normalize(X[testIndex])
        Ytrain, Ytest = Y[trainIndex], Y[testIndex]
        # one hot coding Y
        pYtrain = np.array([np.zeros(len(Ytrain))] * len(classes))
        for i, className in enumerate(Ytrain):
            index = classes.index(className)
            pYtrain[index][i] = 1
        pYtest = np.array([np.zeros(len(Ytest))] * len(classes))
        for i, className in enumerate(Ytest):
            index = classes.index(className)
            pYtest[index][i] = 1
        kFolds.append((Xtrain, pYtrain, Xtest, pYtest))
    return kFolds, first, last



numFolds = 10
kFolds, first, last = processData('wine', numFolds=numFolds) # change wine to vote, cancer, or contraceptive
lamb = 0.0000000001
alph = 1.05
epsilon = 0.00001  # stopping criteria for jcost
numiter = 5000
for Xtrain, Ytrain, Xtest, Ytest in kFolds:
    params = model(Xtrain.T, Ytrain, Xtest.T, Ytest, layerSizes=[first, 50, last], numiter=numiter, lamb=lamb,
                   alph=alph, epsilon=epsilon, testMode=False)
print('Accuracy: ', accuracyy / numFolds)
print('F1: ', f11 / numFolds)

