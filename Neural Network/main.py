import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import pandas as pd

data = pd.read_csv('./datasets/hw3_wine.csv', sep='\t').values
data = pd.read_csv('./datasets/hw3_wine.csv', sep='\t').sample(frac=1).values
Y = data[:, 0]
X = data[:, 1:]
# kf = KFold(n_splits=10)
# folds = kf.split(data)
# for train, test in folds:
#     print(test)
#     print(train)
# print(data)
# train, test = train_test_split(data, test_size=0.2, random_state=10)
# print(train)

kfold = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
folds = kfold.split(X, Y)
for train, test in folds:
    xtrain = X[train]
    ytrain = Y[train]
    xtest = X[test]
    ytrain = Y[test]
print(len(xtrain)+len(xtest))