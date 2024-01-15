# -*- coding:utf-8 -*-
from math import log
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import *


# 训练样本 （这里的训练样本已经被切分为了词条）
def loadDataSet():
    # 定义一个二维列表；整个结构可以看作是一个类似于矩阵的数据结构
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量 1 代表侮辱性的文字 0 代表正常的言论
    return postingList, classVec


"""
函数说明:创建词汇表
1、将文档的单词去重（set）
2、然后保存到list集合中作为词汇表 （后续将样本单词转为词向量会用到）
Parameters:
    dataSet:训练集
Returns:
    list(vocabSet) - 创建的词汇表
['my', 'so', 'buying', 'park', 'help', 'licks', 'dalmation', 'please', 'has', 'ate', 'problems', 'garbage', 'him', 
'maybe', 'take', 'steak', 'love', 'cute', 'stupid', 'I', 'stop', 'is', 'flea', 'quit', 'food', 'worthless', 'to', 
'how', 'dog', 'posting', 'mr', 'not']
"""


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个无序不重复空集，可以使用大括号{} 或 set() 函数来创建一个集合（Set）
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # ｜ 创建两个集合的并集
    return list(vocabSet)  # 集合转为列表


"""
函数说明:根据词汇表将训练样本 转为词向量
Parameters:
    vocabSList:词汇表
    inputSet：训练样本
Returns:
    returnVec - 训练样本转为词向量
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
"""


def setOfWords2Vec(vocabSList, inputSet):
    returnVec = [0] * len(vocabSList)  # 创建一个一行 len(vocabSList) 列；元素值为0的矩阵
    for word in inputSet:
        if word in vocabSList:
            returnVec[vocabSList.index(word)] = 1  # 获取指定单词的索引，并指定索引位置赋值为1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec


"""
函数说明:朴素贝叶斯分类器训练函数 
       求出的结果作为后续分类使用
Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
"""


def trainNB0(trainMatrix, trainGategory):
    numTrainDocs = len(trainMatrix)  # 训练样本数
    numWords = len(trainMatrix[0])  # 样本单词数
    pAbusive = sum(trainGategory) / float(numTrainDocs)  # 训练样本中侮辱性言论所占的比率
    p0Num = ones(numWords)  # 正常言论样本中，增加词条的计数值
    p1Num = ones(numWords)  # 侮辱言论样本中，增加词条的计数值
    p0Demo = 2.0  # 正常言论样本中，所有词条的计数值
    p1Demo = 2.0  # 侮辱言论样本中，所有词条的计数值
    for i in range(numTrainDocs):
        if trainGategory[i] == 1:
            p1Num += trainMatrix[i]  # 侮辱性言论样本中 各单词出现的次数
            p1Demo += sum(trainMatrix[i])  # 侮辱性言论样本中 总共的单词数
        else:
            p0Num += trainMatrix[i]  # 正常言论样本中 各单词出现的次数
            p0Demo += sum(trainMatrix[i])  # 正常言论样本中 各单词出现的次数

    p1Vect = log(p1Num / p1Demo)
    p0Vect = log(p0Num / p0Demo)
    return p0Vect, p1Vect, pAbusive


"""
sum(vec2Classify * p1Vec) + log(pClass1) 
p1Vec本身是对数函数，vec2Classify * p1Vec不为0的就是要预测的特征
sum：这里的sum是对对数函数的sum，变换后也就是各特征概率的相乘
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 自然对数计算 ln(a*b)=ln(a)+ln(b)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    # myVocabList = createVocabList(postingList)
    # print(myVocabList)
    # wordVec = setOfWords2Vec(myVocabList, postingList[0])
    # print(wordVec)
    # trainMat = []  # 训练集
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print(trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # print(p0V)
    # print(p1V)
    # print(pAb)
    testingNB()
