import translators as ts
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd

def getData(file):
    fileSrc = open(file)
    dataset = []
    item = {}
    count = 0
    for index, i in enumerate(fileSrc):
        count+=1
    fileLength = count
    print(fileLength)
    count = 0
    fileSrc = open(file)

    for index, i in enumerate(fileSrc):
        count+=1
        if(index%6 == 0):
            item['source'] = i.replace("\n", "")
        elif(index%6==1):
            item['ref'] = i.replace("\n", "")
        elif(index%6 ==2):
            item['cand'] = i.replace("\n", "")
        elif(index%6 ==3):
            item['qual'] = i.replace("\n", "")
        elif(index%6 ==4):
            if i.replace("\n", "") == 'H':
                item['target'] = 1
            else:
                item['target'] = 0
            if count == fileLength:
                dataset.append(item)            
                item['data'] = item['source'] + ' ' + item['cand']
                item = {}
        elif(index%6 ==5):
            dataset.append(item)            
            item['data'] = item['source'] + ' ' + item['cand']
            item = {}
    return dataset

def getMoreBleuScores(dataset):
    bleuScores2 = np.array([sentence_bleu([x['ref'].split(" ")],x['cand'].split(" "), weights=(0.5, 0.5, 0, 0) ) for x in dataset])
    bleuScores3 = np.array([sentence_bleu([x['ref'].split(" ")],x['cand'].split(" "), weights=(0.33, 0.33, 0.33, 0) ) for x in dataset])
    bleuScores4 = np.array([sentence_bleu([x['ref'].split(" ")],x['cand'].split(" "), weights=(0.25, 0.25, 0.25, 0.25) ) for x in dataset])
    for index,i in enumerate(dataset):
        dataset[index]['2gramBleu'] = bleuScores2[index]
        dataset[index]['3gramBleu'] = bleuScores3[index]
        dataset[index]['4gramBleu'] = bleuScores4[index]
    return dataset

def runBackTranslations(dataset):
    for index, ex in enumerate(dataset):
        chineseTrans = ts.google(ex['cand'],to_language ='zh')
        ex['backTrans'] = ts.google(chineseTrans, to_language ='en')
        ex['backTransGram1'] = sentence_bleu([x['cand'].split(" ")],ex['backTrans'].split(" "), weights=(1,0, 0, 0) )
        ex['backTransGram2'] = sentence_bleu([x['cand'].split(" ")],ex['backTrans'].split(" "), weights=(0.5,0.5, 0, 0) )
        ex['backTransGram3'] = sentence_bleu([x['cand'].split(" ")],ex['backTrans'].split(" "), weights=(0.33,0.33, 0.33, 0) )
        ex['backTransGram4'] = sentence_bleu([x['cand'].split(" ")],ex['backTrans'].split(" "), weights=(0.25,0.25, 0.25, 0.25) )

    return dataset

train = getData("../data/train.txt")
test = getData("../data/test.txt")
train = getMoreBleuScores(train)
test = getMoreBleuScores(test)

trainPD = pd.DataFrame.from_records(train)
trainPD.to_csv('../data/trainProcessed.csv', index=True)  

testPD = pd.DataFrame.from_records(test)
testPD.to_csv('../data/testProcessed.csv', index=True)  
