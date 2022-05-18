import random
import matplotlib.pyplot as plt
from run import load_training_set, load_test_set
from math import log10 as log

class Model:
    def __init__(self, data, alpha=0):
        # without noticing, alpha for Laplace smoothing is zero automatically
        (self.pos, self.neg, self.vocab) = data;
        # freq will record frequency of words in positive and negative comment
        self.freqInPos = {};
        self.freqInNeg = {};
        # probability of this word being positive or negative
        self.probPos = {};
        self.probNeg = {};
        # save alpha for later reference
        self.alpha = alpha
        # train da model
        self.trainModel(alpha);

    def setAlpha(self, alpha):
        self.alpha = alpha;
        self.trainModel(alpha);

    def getPriorProbability(self, className):
        sum = len(self.pos) + len(self.neg);
        if className == 'positive':
            return len(self.pos) / sum;
        if className == 'negative':
            return len(self.neg) / sum;

    def getDataInfo(self):
        print("Number of positive training instances: ", len(self.pos))
        print("Number of negative training instances: ", len(self.neg))
        print("Vocabulary size: ", len(self.vocab))

    def getVocab(self) -> list[str]:
        return self.vocab;

    def trainModel(self, alpha=0):
        # calculate frequency
        for text in self.pos:
            for word in text:
                if word in self.freqInPos:
                    self.freqInPos[word] += 1;
                else:
                    self.freqInPos[word] = 1;
        for text in self.neg:
            for word in text:
                if word in self.freqInNeg:
                    self.freqInNeg[word] += 1;
                else:
                    self.freqInNeg[word] = 1;
        self.laplaceSmoothing(alpha);

    def laplaceSmoothing(self, alpha):
        sumFreqPos = 0;
        sumFreqNeg = 0;
        # compute how many words appear in positive/negative reviews.
        for word in self.vocab:
            sumFreqPos += self.freqInPos.get(word, 0);
            sumFreqNeg += self.freqInNeg.get(word, 0);
        for word in self.vocab:
            # frequency of thi word appears in a positive/negative review. say zero if it never appears.
            pos = self.freqInPos.get(word, 0);
            neg = self.freqInNeg.get(word, 0);
            self.probPos[word] = (pos + alpha) / (sumFreqPos + alpha * len(self.vocab));
            self.probNeg[word] = (neg + alpha) / (sumFreqNeg + alpha * len(self.vocab));

    def getRandomClass(self):
        i = random.randint(0, 1);
        return 'positive' if i == 0 else 'negative';

    def predictWithoutLog(self, text: list[str]) -> str:
        productPos = 1;
        productNeg = 1;
        for word in text:
            productPos *= self.probPos.get(word, 0);
            productNeg *= self.probNeg.get(word, 0);
        decision = productNeg * self.getPriorProbability('negative') - productPos * self.getPriorProbability(
            'positive');
        return 'negative' if decision > 0 else 'positive' if decision < 0 else self.getRandomClass();

    def predictWithLog(self, text: list[str]) -> str:
        productPos = 0;
        productNeg = 0;
        for word in text:
            productPos += log(self.probPos.get(word, 0) if self.probPos.get(word, 0) != 0 else 1)
            productNeg += log(self.probNeg.get(word, 0) if self.probNeg.get(word, 0) != 0 else 1)
        decision = log(self.getPriorProbability('negative')) + productNeg - log(
            self.getPriorProbability('positive')) - productPos
        return 'negative' if decision > 0 else 'positive' if decision < 0 else self.getRandomClass();

    def predict(self, text: list[str]) -> str:
        # predictWithLog can be used even when after laplace smoothing since if a zero probability can never happen
        return self.predictWithLog(text);

def runMeForQuestion1():
    trainingData = load_training_set(0.2, 0.2)
    testData = load_test_set(0.2, 0.2)
    model = Model(trainingData)
    negativeCount = 0;
    negativeTestSetSize = len(testData[1]);
    positiveCount = 0;
    positiveTestSetSize = len(testData[0]);
    for text in testData[0]:
        if model.predictWithoutLog(text) == 'positive':
            positiveCount += 1;
    for text in testData[1]:
        if model.predictWithoutLog(text) == 'negative':
            negativeCount += 1;
    accuracy = (positiveCount / positiveTestSetSize + negativeCount / negativeTestSetSize) / 2;
    confusionMatrix = {
        'TP': positiveCount,
        'FN': positiveTestSetSize - positiveCount,
        'FP': negativeTestSetSize - negativeCount,
        'TN': negativeCount
    }
    precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP']);
    recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN']);
    print('Predict without using log trick')
    print('Accuracy: ', accuracy);
    print('Confusion Matrix: ', confusionMatrix);
    print('Precision: ', precision);
    print('Recall: ', recall)
    print('-----------------------------------------------')

    negativeCount = 0;
    positiveCount = 0;
    for text in testData[0]:
        if model.predictWithLog(text) == 'positive':
            positiveCount += 1;
    for text in testData[1]:
        if model.predictWithLog(text) == 'negative':
            negativeCount += 1;
    accuracy = (positiveCount / positiveTestSetSize + negativeCount / negativeTestSetSize) / 2;
    confusionMatrix = {
        'TP': positiveCount,
        'FN': positiveTestSetSize - positiveCount,
        'FP': negativeTestSetSize - negativeCount,
        'TN': negativeCount
    }
    precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP']);
    recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN']);
    print('Predict with using log trick')
    print('Accuracy: ', accuracy);
    print('Confusion Matrix: ', confusionMatrix);
    print('Precision: ', precision);
    print('Recall: ', recall)
    print('-----------------------------------------------')
# runMeForQuestion1();

def runMeForQuestion2():
    trainingData = load_training_set(0.2, 0.2)
    testData = load_test_set(0.2, 0.2)
    model = Model(trainingData)

    def accuracyByAlpha(alpha):
        model.setAlpha(alpha)
        negativeCount = 0;
        negativeTestSetSize = len(testData[1]);
        positiveCount = 0;
        positiveTestSetSize = len(testData[0]);
        for text in testData[0]:
            if model.predict(text) == 'positive':
                positiveCount += 1;
        for text in testData[1]:
            if model.predict(text) == 'negative':
                negativeCount += 1;
        accuracy = (positiveCount / positiveTestSetSize + negativeCount / negativeTestSetSize) / 2;
        confusionMatrix = {
            'TP': positiveCount,
            'FN': positiveTestSetSize - positiveCount,
            'FP': negativeTestSetSize - negativeCount,
            'TN': negativeCount
        }
        precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP']);
        recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN']);
        return {
            'accuracy': accuracy,
            'confusionMatrix': confusionMatrix,
            'precision': precision,
            'recall': recall,
        }

    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    y = []
    x = [];
    for i in alphas:
        trainRes = accuracyByAlpha(i);
        y.append(trainRes['accuracy']);
        print('Alpha: ', i);
        print('Accuracy: ', trainRes['accuracy']);
        print('Confusion Matrix: ', trainRes['confusionMatrix']);
        print('Precision: ', trainRes['precision']);
        print('Recall: ', trainRes['recall'])
        print('-----------------------------------------------')
        x.append(log(i));
    plt.plot(x, y);
    plt.show();
runMeForQuestion2();

def runMeForQuestion3():
    # let's use alpha = 10
    trainingData = load_training_set(1, 1)
    testData = load_test_set(0.2, 0.2)
    model = Model(trainingData)

    def accuracyByAlpha(alpha):
        model.setAlpha(alpha)
        negativeCount = 0;
        negativeTestSetSize = len(testData[1]);
        positiveCount = 0;
        positiveTestSetSize = len(testData[0]);
        # use posterior log-probabilities
        for text in testData[0]:
            if model.predictWithLog(text) == 'positive':
                positiveCount += 1;
        for text in testData[1]:
            if model.predictWithLog(text) == 'negative':
                negativeCount += 1;
        accuracy = (positiveCount / positiveTestSetSize + negativeCount / negativeTestSetSize) / 2;
        confusionMatrix = {
            'TP': positiveCount,
            'FN': positiveTestSetSize - positiveCount,
            'FP': negativeTestSetSize - negativeCount,
            'TN': negativeCount
        }
        precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP']);
        recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN']);
        return {
            'accuracy': accuracy,
            'confusionMatrix': confusionMatrix,
            'precision': precision,
            'recall': recall,
        }

    res = accuracyByAlpha(10);
    print('Alpha: ', 10);
    print('Accuracy: ', res['accuracy']);
    print('Confusion Matrix: ', res['confusionMatrix']);
    print('Precision: ', res['precision']);
    print('Recall: ', res['recall'])
    print('-----------------------------------------------')
# runMeForQuestion3()

def runMeForQuestion4():
    #50% training set, 100% test set
    trainingData = load_training_set(0.5, 0.5)
    testData = load_test_set(1, 1)
    model = Model(trainingData)

    def accuracyByAlpha(alpha):
        model.setAlpha(alpha)
        negativeCount = 0;
        negativeTestSetSize = len(testData[1]);
        positiveCount = 0;
        positiveTestSetSize = len(testData[0]);
        # use posterior log-probabilities
        for text in testData[0]:
            if model.predictWithLog(text) == 'positive':
                positiveCount += 1;
        for text in testData[1]:
            if model.predictWithLog(text) == 'negative':
                negativeCount += 1;
        accuracy = (positiveCount / positiveTestSetSize + negativeCount / negativeTestSetSize) / 2;
        confusionMatrix = {
            'TP': positiveCount,
            'FN': positiveTestSetSize - positiveCount,
            'FP': negativeTestSetSize - negativeCount,
            'TN': negativeCount
        }
        precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP']);
        recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN']);
        return {
            'accuracy': accuracy,
            'confusionMatrix': confusionMatrix,
            'precision': precision,
            'recall': recall,
        }

    res = accuracyByAlpha(10);
    print('Alpha: ', 10);
    print('Accuracy: ', res['accuracy']);
    print('Confusion Matrix: ', res['confusionMatrix']);
    print('Precision: ', res['precision']);
    print('Recall: ', res['recall'])
    print('-----------------------------------------------')
# runMeForQuestion4();

def runMeForQuestion6():
    #10% positive, 50% negative on training set. 100% for test set
    trainingData = load_training_set(0.1, 0.5)
    testData = load_test_set(1, 1)
    model = Model(trainingData)

    def accuracyByAlpha(alpha):
        model.setAlpha(alpha)
        negativeCount = 0;
        negativeTestSetSize = len(testData[1]);
        positiveCount = 0;
        positiveTestSetSize = len(testData[0]);
        # use posterior log-probabilities
        for text in testData[0]:
            if model.predictWithLog(text) == 'positive':
                positiveCount += 1;
        for text in testData[1]:
            if model.predictWithLog(text) == 'negative':
                negativeCount += 1;
        accuracy = (positiveCount / positiveTestSetSize + negativeCount / negativeTestSetSize) / 2;
        confusionMatrix = {
            'TP': positiveCount,
            'FN': positiveTestSetSize - positiveCount,
            'FP': negativeTestSetSize - negativeCount,
            'TN': negativeCount
        }
        precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP']);
        recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN']);
        return {
            'accuracy': accuracy,
            'confusionMatrix': confusionMatrix,
            'precision': precision,
            'recall': recall,
        }

    res = accuracyByAlpha(10);
    print('Alpha: ', 10);
    print('Accuracy: ', res['accuracy']);
    print('Confusion Matrix: ', res['confusionMatrix']);
    print('Precision: ', res['precision']);
    print('Recall: ', res['recall'])
    print('-----------------------------------------------')
# runMeForQuestion6();