from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# data = datasets.load_digits()
# X, Y = data["data"], data["target"]
# print(X[0])
# print(Y[0])

import math
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


def normalize(df, exc):
    index = 0
    res = df.copy();
    for feature in df.columns:
        if index != exc:
            min = df[feature].min();
            max = df[feature].max();
            res[feature] = (df[feature] - min) / (max - min);
        index += 1
    return res;


class Instance:
    def __init__(self, row, classCol):
        self.className = row[classCol]
        self.vector = []
        for i in range(len(row)):
            if i != classCol:
                self.vector.append(row[i])


class ProcessedData:
    def __init__(self, classified: dict):
        self.classified = classified
        # pool contains unclassified data.
        self.pool = [];
        self.classNames = [];
        self.shuffle()
        for className in self.classNames:
            for instance in self.classified[className]:
                self.pool.append(instance);
        # fold holds k ProcessedData object.
        random.shuffle(self.pool);
        self.folds = [];

    def shuffle(self):
        # go to each class in classified and shuffle.
        for key in self.classified.keys():
            random.shuffle(self.classified[key]);
            self.classNames.append(key)

    def fold(self, k):
        for time in range(k):
            fold = {}
            for className in self.classNames:
                size = int(len(self.classified[className]) / k);
                start = size * time;
                end = min(size * (time + 1), len(self.classified[className]) - 1);
                fold.setdefault(className, [])
                for i in range(start, end + 1):
                    # fold.append(self.classified[className][i])
                    fold.get(className).append(self.classified[className][i])
            self.folds.append(ProcessedData(fold))

    def combineExcept(self, ex):
        count = 0
        res = {}
        for className in self.classNames:
            res.setdefault(className, []);
        for fold in self.folds:
            if count != ex:
                for className in self.classNames:
                    for instance in fold.classified[className]:
                        res[className].append(instance);
            count += 1
        return ProcessedData(res);

    def prepareDataToTrainForest(self, ex):
        # ith fold hold for testing, the other k-1 for training
        return {
            'test': self.folds[ex],
            'train': self.combineExcept(ex)
        }

    def sampleDataWithReplacement(self):
        # go to classified and sample randomly with replacement.
        res = {};
        for className in self.classNames:
            res.setdefault(className, []);
        ran = np.random.choice(self.pool, size=len(self.pool), replace=True)
        for instance in ran:
            res[instance.className].append(instance);
        return ProcessedData(res);

    def getValuesOfAttribute(self, attr):
        res = set()
        for instance in self.pool:
            res.add(instance.vector[attr])
        return list(res)

    def isSameClass(self):
        # return className if there is only that class Name,
        count = 0;
        last = -1
        for className in self.classNames:
            if len(self.classified[className]) > 0:
                count += 1
                last = className
        return last if count == 1 else -1

    def getMajority(self):
        maxClass = 0
        for className in self.classNames:
            if len(self.classified[className]) > len(self.classified[className]):
                maxClass = className
        return maxClass


def processData(dataName):
    # receive raw data. classify these instances into different array according to their classes. return a jagged array.
    # return {cn1: [instances], ..., cnn: [instances of cnn]}
    match dataName:
        case 'wine':
            nameFile = 'hw3_wine.csv'
            className = [1, 2, 3]
            classCol = 0
            sep = '\t'
        case 'vote':
            nameFile = 'hw3_house_votes_84.csv'
            className = [0, 1]
            classCol = 16
            sep = ','
        case 'parkinson':
            nameFile = 'parkinsons.csv'
            className = [0, 1]
            classCol = 22
            sep = ','
        case 'titanic':
            nameFile = 'titanic.csv'
            className = [0, 1]
            classCol = 0
            sep = ','
        case 'loan':
            nameFile = 'loan.csv'
            className = [0, 1]
            classCol = 11
            sep = ','
        case 'digits':
            nameFile = 'digits.csv'
            className = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            classCol = 64
            sep = ','
        case 'cmc':
            nameFile = 'cmc.csv'
            className = [0, 1, 2]
            classCol = 9
            sep = ','
        case _:
            nameFile = 'hw3_cancer.csv'
            className = [0, 1]
            classCol = 9
            sep = '\t'

    df = pd.read_csv('./datasets/'+nameFile, sep=sep)
    if nameFile == 'titanic.csv':
        # delete the third col
        df = df.drop(columns=['Name'])
        df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
    if nameFile == 'loan.csv':
        df['Gender'] = np.where(df['Gender'] == 'Male', 1, 0)
        df['Loan_Status'] = np.where(df['Loan_Status'] == 'Y', 1, 0)
        df['Education'] = np.where(df['Education'] == 'Graduate', 1, 0)
        df['Self_Employed'] = np.where(df['Self_Employed'] == 'Yes', 1, 0)
        df['Property_Area'] = np.where(df['Property_Area'] == 'Urban', 1, 0)
        df['Married'] = np.where(df['Married'] == 'Yes', 1, 0)
        df = df.drop(columns=['Loan_ID'])
    df = df.astype(float)
    df = normalize(df, exc=classCol);
    classified = {};
    for cn in className:
        classified.setdefault(cn, [])
    for _, row in df.iterrows():
        vector = [];
        for _, val in row.items():
            vector.append(val)
        instance = Instance(vector, classCol);
        classified.get(row[classCol]).append(instance);
    return ProcessedData(classified)


def foldTheData(k):
    # receive the classified data from processData, then create k folds from it with the same ratio of classes as in
    # the input
    return [{'train': [], 'test': []}, {'train': [], 'test': []}]


class Node:
    def __init__(self, classifier=None, attribute=None, label=None, edge: list = []):
        # node will test attribute att of an instance. depend on the result, it will continue going to deeper node.
        # classifier
        self.label = label;
        self.attribute = attribute;
        self.neighbor = {};
        self.classifier = classifier if classifier else lambda x: x
        # classifier receives an instance and return the edge this instance should be passed to
        # value returned by classifier is always a key of neighbor
        # in case of categorical attribute, return the input, else hash it
        if len(edge) != 0:
            for e in edge:
                self.neighbor.setdefault(e, None)

    def addEdge(self, val, node):
        # if prediction is of this val, go to this node
        self.neighbor[val] = node;

    def test(self, instance: Instance):
        if self.label is not None: return self.label;  # leaf node
        # test attribute att of this instance
        valAfterTest = self.classifier(instance.vector[self.attribute]);
        # next node
        return self.neighbor[valAfterTest].test(instance);


class RandomForest:
    def __init__(self, ntree: int, Dtrain: ProcessedData, Dtest: ProcessedData, classNames: list):
        # receive training data and the hyperparameter ntree to create a random forest. Test data is later used to test
        # the forest's performance
        self.ntree = ntree
        self.Dtrain = Dtrain
        self.Dtest = Dtest
        self.classNames = classNames
        self.attributes = list(range(0, len(Dtrain.classified[1][0].vector)));
        self.committee = []  # used to save trees.
        self.train()
        # create ntree bootstraps by sampling with replacement from Dtrain, then train ntree.

    def train(self):
        # create ntree using data sampling from Dtrain.
        for i in range(self.ntree):
            # sample with replacement a training set with the same size as Dtrain from Dtrain.
            self.committee.append(
                self.decisionTree(Dtrain=self.Dtrain.sampleDataWithReplacement(), L=self.attributes.copy()));

    def testAccuracy(self):
        # take the test set out. perform test
        def confusionMatrix(positiveClass):
            res = {
                'TP': 1,
                'FP': 1,
                'TN': 1,
                'FN': 1,
            }
            n = len(self.Dtest.pool)
            for className in self.classNames:
                for inst in self.Dtest.classified[className]:
                    predictedClass = self.testInstance(inst);
                    if className == positiveClass:
                        if predictedClass == inst.className:
                            res['TP'] += 1
                        else:
                            res['FP'] += 1
                    else:
                        if predictedClass == inst.className:
                            res['TN'] += 1
                        else:
                            res['FN'] += 1
            accuracy = (res['TP'] + res['TN']) / n
            precision = res['TP'] / (res['TP'] + res['FP'])
            recall = res['TP'] / (res['TP'] + res['FN'])
            F1 = res['TP'] / (res['TP'] + (res['FP'] + res['FN']) / 2)
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'F1': F1
            }

        final = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'F1': 0
        }
        for className in self.classNames:
            res = confusionMatrix(className)
            for key in final.keys():
                final[key] += res[key]
        for key in final.keys():
            final[key] /= len(self.classNames)
        return final

    def testInstance(self, instance):
        countClass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for tree in self.committee:
            countClass[tree.test(instance)] += 1
        predictedClass = countClass.index(max(countClass))
        return predictedClass

    def decisionTree(self, Dtrain: ProcessedData, L: list) -> Node:
        # use Dtrain provided in the train function to train tree
        # create a function to decide label for each instance. This should be trivial in case of categorical data, just
        # by looking at the category of this attribute. For numerical one, this function should decide wisely
        # L is a list of attribute left to consider
        isSameClass = Dtrain.isSameClass();
        if isSameClass >= 0: return Node(label=isSameClass)

        def infoGain(l1):
            def entropy(arr1):
                # calculate the probability and entropy
                ps = []
                summ = 0
                for i in arr1: summ += i
                if summ == 0: return 1
                for i in arr1:
                    ps.append(i / summ)
                final = 0
                for p in ps:
                    if p != 0:
                        final += -p * math.log2(p)
                return final

            V = {}
            categorical = False
            values = [0, 1]
            for cn in values:
                V.setdefault(cn, [0] * len(Dtrain.classNames))

            # V[val, count of val in each class]
            # V contains values of v,
            # create a classifier for values of V
            # if V is categorical:

            def classifier(x):
                if categorical:
                    return x
                avg = 0
                for ins in Dtrain.pool:
                    avg += ins.vector[l1]
                avg /= len(Dtrain.pool)
                return 0 if x < avg else 1

            for className1 in Dtrain.classNames:
                for instance1 in Dtrain.classified[className1]:
                    x = classifier(instance1.vector[l1])
                    V[x][className1] += 1
            info = 0
            for arr in V.values():
                info += (sum(arr) / len(Dtrain.pool)) * entropy(arr)
            return {
                'info': info,
                'edge': [0, 1],
                'classifier': classifier,
            }

        # attribute with the highest info
        A = -1
        lowest = 9
        for l in L:
            point = infoGain(l)
            if point['info'] < lowest:
                A = l
                lowest = point['info']
                classifier = point['classifier']
                edge = point['edge']
        if A == -1: return Node(label=Dtrain.getMajority())
        N = Node(attribute=A, classifier=classifier, edge=edge)
        L.remove(A)
        # only for categorical values.
        V = edge;
        for v in V:
            res = {}
            update = False
            for className in self.classNames:
                res.setdefault(className, [])
            # empty attribute
            for instance in Dtrain.pool:
                if classifier(instance.vector[A]) == v:
                    res[instance.className].append(instance)
                    update = True
            if not update: return Node(label=Dtrain.getMajority())
            N.addEdge(v, self.decisionTree(ProcessedData(res), L.copy()))
        return N


# hyperparameter
K = 3
data = 'cmc'


# train random forest
def trainRandomForests(K, ntree, data):
    pData = processData(data);
    # fold
    pData.fold(K)
    randomForests = []
    for i in range(K):
        expKFold = pData.prepareDataToTrainForest(i);
        randomForests.append(
            RandomForest(ntree=ntree, Dtrain=expKFold['train'], Dtest=expKFold['test'], classNames=pData.classNames))
    final = {'accuracy': 0, 'precision': 0, 'recall': 0, 'F1': 0}
    for rf in randomForests:
        res = rf.testAccuracy()
        for key in final.keys():
            final[key] += res[key]
    for key in final.keys():
        final[key] /= len(randomForests)
    return final


def plotMetricsGraph():
    acc = []
    pre = []
    rec = []
    f1 = []
    ntrees = [1, 5, 10, 20, 30, 40, 50]
    for i in ntrees:
        res = trainRandomForests(K, i, data)
        acc.append(res['accuracy'])
        pre.append(res['precision'])
        rec.append(res['recall'])
        f1.append(res['F1'])
    plt.plot(ntrees, pre)
    plt.ylabel('Precision')
    plt.xlabel('Number of trees')
    plt.show()

    plt.plot(ntrees, acc)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of trees')
    plt.show()

    plt.plot(ntrees, rec)
    plt.ylabel('Recall')
    plt.xlabel('Number of trees')
    plt.show()

    plt.plot(ntrees, f1)
    plt.ylabel('F1 Score')
    plt.xlabel('Number of trees')
    plt.show()


plotMetricsGraph()
