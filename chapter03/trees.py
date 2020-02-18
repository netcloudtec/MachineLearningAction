# -*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from math import log
import treePlotter

# 定义一个数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算信息熵 （标签的类别越多值越大，当熵的值为0时 标签的类别都是一样的）
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 统计数据集的样本数
    labelCount = {}  # 定义一个字典 统计各个label的数目
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key] / numEntries)  # 标签出现的概率
        shannonEnt -= prob * log(prob, 2)  # 根据标签出现的概率，计算信息熵
    return shannonEnt


# 划分数据集 axis值为特征的索引，value是特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 列表切片
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选取最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 获取样本的特征值数目
    baseEntropy = calcShannonEnt(dataSet)  # 获取数据集的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 获取第n个特征的的特征值可能情况存储到set中
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 拆分数据 第n个特征 其特征值为 value
            prob = len(subDataSet) / float(len(dataSet))  # 拆分后数据子集 占 整个数据集的比例
            newEntropy += prob * calcShannonEnt(subDataSet)  # 获取新的熵
        infoGain = baseEntropy - newEntropy  # 整个数据集标签的熵 - 指定特征值标签的熵 （结果值越大越好）
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  #
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# classList 标签列表 返回标签数目最多的那个标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树函数
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 遍历样本标签 存储在列表中
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 返回最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选取最好的数据划分方式 返回特征的索引值
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  #
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 获取第n个特征的特征值
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 决策树分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 决策树存储 python2  python3 修改为
# def storeTree(inputTree, filename):
#     import pickle
#     fw = open(filename, 'w')
#     pickle.dump(inputTree, fw)
#     fw.close()
# python3.7 API
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw, 0)
    fw.close()


# Python2 API
# def grabTree(filename):
#     import pickle
#     fr = open(filename)
#     return pickle.load(fr)
# Python3 API
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


# 决策树预测隐形眼镜类型
def predictAction(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()  # 创建数据集
    shannonEnt = calcShannonEnt(dataSet)  # 计算数据集熵
    print(shannonEnt)
    retDataSet = splitDataSet(dataSet, 0, 1)  # 拆分数据集
    print(retDataSet)
    bestFeature = chooseBestFeatureToSplit(dataSet)  # 选取最好的数据集划分方式
    print(bestFeature)
    # mytree = createTree(dataSet, labels)
    mytree = treePlotter.retrieveTree(0)
    print(mytree)
    classRet = classify(mytree, labels, [1, 1])
    print(classRet)
    storeTree(mytree, 'classifierStorage.txt')  # 保存决策树
    print(grabTree('classifierStorage.txt'))  # 读取决策树
    # 预测隐形眼镜类型
    lensesTree = predictAction("/Users/yangshaojun/python_workspace/chapter03/dataset/lenses.txt")
    print(lensesTree)
    # 绘制树形图
    treePlotter.createPlot(lensesTree)
