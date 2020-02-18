# -*- coding:utf-8 -*-
from math import log
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import *
import re
import random



# 训练样本 （这里的训练样本已经被切分为了词条）
def loadDataSet():
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
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # ｜ 创建两个集合的并集
    return list(vocabSet)


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
    returnVec = [0] * len(vocabSList)  # 创建一个一行 len(vocabSList) 列 元素值为0的矩阵
    for word in inputSet:
        if word in vocabSList:
            returnVec[vocabSList.index(word)] = 1  # 获取指定单词的索引
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


# 垃圾邮件分类
"""
函数说明:接收一个大字符串并将其解析为字符串列表
Parameters:
    bigString
Returns:
    无
"""
def textParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W+', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写

# 使用朴素贝叶斯进行垃圾邮件过滤
def spamTest():
    docList = [];
    classList = [];
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open('/Users/yangshaojun/python_workspace/chapter04/email/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('/Users/yangshaojun/python_workspace/chapter04/email/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet = list(range(50));
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = [];
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print(myVocabList)
    wordVec = setOfWords2Vec(myVocabList, postingList[0])
    print(wordVec)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print(p0V)
    print(p1V)
    print(pAb)
    testingNB()
    spamTest()
