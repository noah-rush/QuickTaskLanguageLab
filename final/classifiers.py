import math
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from matplotlib import pyplot as plt
testPD = pd.read_csv("../data/testProcessed.csv")
trainPD = pd.read_csv("../data/trainProcessed.csv")


def naiveBayes(bagOfWords, targets, toPredict, qual):

    humanRate = np.sum(targets) / len(targets)
    
    humanIndexes = np.argwhere(targets==1).ravel()
    machineIndexes = np.argwhere(targets==0).ravel()
    
    humanTexts = bagOfWords[humanIndexes]
    machineTexts = bagOfWords[machineIndexes]
    
    humanProbs = np.mean(humanTexts, axis =0) 
    machineProbs = np.mean(machineTexts, axis =0) 
    
    humanProbs = humanProbs.clip(1e-14, 1-1e-14)
    machineProbs = machineProbs.clip(1e-14, 1-1e-14)
    
#     Prediction
    logpyHuman = math.log(humanRate)
    logpyMachine= math.log(1 - humanRate)
    
    logpxyHuman = toPredict * np.log(humanProbs) + (1-toPredict) * np.log(1-humanProbs)
    logpxyMachine = toPredict * np.log(machineProbs) + (1-machineProbs) * np.log(1-machineProbs)

    logpyxHuman= logpxyHuman.sum(axis=1) + logpyHuman
    logpyxMachine = logpxyMachine.sum(axis=1) + logpyMachine
    
    logpyxHuman = logpyxHuman+np.average(qual)
    logpyxMachine  = logpyxMachine+(1 - np.average(qual))
    
    preds = logpyxHuman > logpyxMachine
    return preds


def logisticReg(X, testX, targets, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C).fit(X, targets)
    logisticPreds = clf.predict(testX)
    return logisticPreds


def calculateF1(preds, trues, flip = False):
    tps = 0 
    fps = 0
    fns = 0
    if flip:
        for index, i in enumerate(preds):
            if not i and 0 == trues[index]:
                tps += 1
            if not i and 0 != trues[index]:    
                fps += 1
            if i and 0 == trues[index]:
                fns += 1
    else:
        for index, i in enumerate(preds):
            if i and 1 == trues[index]:
                tps += 1
            if i and 1 != trues[index]:    
                fps += 1
            if not i and 1 == trues[index]:
                fns += 1
    f1 = tps/(tps + 0.5*(fps + fns))
    return f1


f1s = []
logF1s = []
for nGrams in range(1, 5):
    f1s.append([])
    logF1s.append([])
    for i in range(1,100):
        M = i
        trainCorpus = trainPD['data'].values.astype('U')
        vectorizer = CountVectorizer(min_df=M, binary=True, ngram_range = (1,nGrams))
        X = vectorizer.fit_transform(trainCorpus)

        targets = trainPD['target'].to_numpy()

        testCorpus = testPD['data'].values.astype('U')
        testVectorizer = CountVectorizer(vocabulary = vectorizer.get_feature_names(),min_df=M, binary=True)
        testVector = testVectorizer.fit_transform(testCorpus)

        features =['qual', '2gramBleu', '3gramBleu', '4gramBleu']

        X = X.toarray()
        Xfeatures = np.append(X, trainPD[features].to_numpy(), axis=1)
        testX = testVector.toarray()
        testXfeatures = np.append(testX, testPD[features].to_numpy(), axis=1)

        preds = naiveBayes(Xfeatures, targets, testXfeatures, testPD[features].to_numpy().astype(np.float) )

        humanF1 = calculateF1(preds, testPD['target'].to_numpy())
        machineF1 = calculateF1(preds, testPD['target'].to_numpy(), True)
        avgF1 = (humanF1 + machineF1) / 2

        f1s[nGrams-1].append(avgF1)

        preds = logisticReg(Xfeatures,testXfeatures,targets, 0.1)

        humanF1 = calculateF1(preds, testPD['target'].to_numpy())
        machineF1 = calculateF1(preds, testPD['target'].to_numpy(), True)
        avgF1 = (humanF1 + machineF1) / 2

        logF1s[nGrams-1].append(avgF1)
        print(i)


plt.figure()
for index,i in enumerate(f1s):
    plt.plot(np.array(range(1,100)), np.array(i), label = "ngram " + str(index))
plt.legend()
plt.title("Naive Bayes Minimum Document Frequency Vs. Average F1 score")
plt.savefig("NaiveBayes.png")


plt.figure()
for index,i in enumerate(logF1s):
    plt.plot(np.array(range(1,100)), np.array(i), label = "ngram " + str(index))
plt.legend()
plt.title("Logistic Regression Minimum Document Frequency Vs. Average F1 score")
plt.savefig("LogReg.png")


neighborsF1s = []
targets = trainPD['target'].to_numpy()

features =['qual', '2gramBleu', '3gramBleu', '4gramBleu']
for i in range(1,201):
    X = trainPD[features]
    testX=testPD[features]

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X, targets)

    preds = neigh.predict(testX)
    # KNeighborsClassifier(...)
    humanF1 = calculateF1(preds, testPD['target'].to_numpy())
    machineF1 = calculateF1(preds, testPD['target'].to_numpy(), True)

    # Human Score F1

    avgF1 = (humanF1 + machineF1) / 2
    neighborsF1s.append(avgF1)

plt.figure()
plt.plot(np.array(range(1,201)), np.array(neighborsF1s))
plt.title("KNN value of K Vs. Average F1 score")
plt.savefig("KNN.png")


M = 12
trainCorpus = trainPD['data'].values.astype('U')
vectorizer = CountVectorizer(min_df=M, binary=True, ngram_range = (1,1))
X = vectorizer.fit_transform(trainCorpus)

targets = trainPD['target'].to_numpy()
testCorpus = testPD['data'].values.astype('U')
testVectorizer = CountVectorizer(vocabulary = vectorizer.get_feature_names(),min_df=M, binary=True)
testVector = testVectorizer.fit_transform(testCorpus)

features =['qual', '2gramBleu', '3gramBleu', '4gramBleu']

X = X.toarray()
Xfeatures = np.append(X, trainPD[features].to_numpy(), axis=1)
testX = testVector.toarray()
testXfeatures = np.append(testX, testPD[features].to_numpy(), axis=1)

preds = naiveBayes(Xfeatures, targets, testXfeatures, testPD[features].to_numpy().astype(np.float) )

humanF1 = calculateF1(preds, testPD['target'].to_numpy())
machineF1 = calculateF1(preds, testPD['target'].to_numpy(), True)
avgF1 = (humanF1 + machineF1) / 2

print(avgF1)
