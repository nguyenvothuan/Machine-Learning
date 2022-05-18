
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

originals = []
for i in range(20):
    reader = csv.reader(open('iris.csv'));
    original = [];
    for row in reader:
        original.append(row);
    originals.append(original);
datas =[]
tests = []
#originals[i] = datas[i] + tests[i];

MAX_K = 100

#custom object
class Instance:
    def __init__(self,distance,index):
        self.distance = distance;
        self.index = index;
class DataSet:
    def __init__(self, training, testing):
        self.training = training;
        self.testing = testing;
#shuffle data. Take the first 80% for training and 20% for testing. Always call this after running the above functions

def countType():
    a = 0;b=0;c=0
    for row in original:
        if row[4]=='Iris-setosa':
            a+=1;
        if row[4]=='Iris-versicolor':
            b+=1;
        if row[4]=='Iris-virginica':
            c+=1;
    return [a,b,c];
# print(countType())


def prepareData():
    for i in range(len(originals)):
        np.random.shuffle(originals[i]);
        datas.append(originals[i][:120]);
        tests.append(originals[i][120:]);

def normalizeData(setId):
    test = tests[setId];
    data = datas[setId];
    maxCellByColumns = [];
    for i in range(4):
        maxVal = -1
        for j in range(len(data)):
            maxVal = max(float(data[j][i]), maxVal);
        maxCellByColumns.append(maxVal);
    return maxCellByColumns
# divideMe = normalizeData()

#a and b are two instances of iris
def euclideanDistance(a, b):
    sum = 0;
    for i in range(4):
        sum += (float(a[i])-float(b[i]))**2
    return math.sqrt(sum);

# print(euclideanDistance(data[99], data[1]))

#prepare distances for row ith. distance from its self are zero. save the index of the current one and distance
def prepareDistance(rowInd, setId, isTesting = False, ):
    test = tests[setId];
    data = datas[setId];
    distances = [];
    for i in range(len(data)):
        distances.append(Instance(euclideanDistance((test if isTesting else data)[rowInd], data[i]), i));
    distances.sort(key = lambda x: x.distance)
    (test if isTesting else data)[rowInd].append(distances);

def prepareDistances(setId, isTesting = False):
    test = tests[setId];
    data = datas[setId];
    for i in range(len(test if isTesting else data)):
        prepareDistance(i, setId, isTesting = isTesting);

# prepareDistances();
# prepareDistances(isTesting=True);

# print(test[0][5][30].distance)


def kNN(k, instance, setId):
    #get the first nearest k instances to data, and based on the vote
    test = tests[setId];
    data = datas[setId];
    setosa = 0; versicolor=0; virginica =0
    #get the first k nearest instances
    for i in range(1,k+1): #exclude self, since it is zero and useless
         index = instance[5][i].index
         if data[index][4]=='Iris-setosa': setosa+=1;
         elif data[index][4]=='Iris-versicolor': versicolor+=1;
         else: virginica+=1;
    decision = max(setosa, versicolor, virginica);
    if decision == setosa: decision = 'Iris-setosa';
    elif decision == versicolor: decision = 'Iris-versicolor';
    else: decision = 'Iris-virginica';
    # print('setosa: ', setosa, '; versicolor: ', versicolor, 'virginica: ', virginica)
    return instance[4] == decision;

def computeAccuracyTraining(k, setId):
    data = datas[setId];
    countTrue = 0
    for i in range(len(data)): #compute accuracy for each k on the dataset.
        if kNN(k, data[i], setId): countTrue+=1;
    return countTrue / len(data)#how many true result returned on the whole training

def computeAccuracyTesting(k, setId):
    test = tests[setId];
    data = datas[setId];
    countTrue = 0
    for i in range(len(test)):
        if kNN(k, test[i], setId): countTrue+=1;
    return countTrue / len(test)#how many true guesses based on the true value

def plot(fun):
    # k =np.arange(1,MAX_K+1,2);
    # ys=[];
    # std =[];
    # y=[];
    k=np.arange(1,MAX_K+1, 2);
    y =[];
    prepareData();
    for i in range(len(k)):
        y.append([]);
    for time in range(20):
        curY = [];
        prepareDistances(time);
        prepareDistances(time, True);
        for i in k:
            curY.append(fun(i,time));
        for i in range(len(k)):
            y[i].append(curY[i]);
    avg = [];
    std = [];
    for arr in y:
        avg.append(np.average(arr));
        std.append(np.std(arr));
    # print(avg);
    plt.scatter(k, avg);
    plt.errorbar(k,avg,yerr=std);
    plt.show();

plot(computeAccuracyTesting)
#plot(computeAccuracyTraining)