# -*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 使用numpy科学计算包模块中的array创建矩阵
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 矩阵的大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # https://www.cnblogs.com/zibu1234/p/4210521.html
    sqDiffMat = diffMat ** 2
    sqlDistances = sqDiffMat.sum(axis=1)  # 对于二维数组 axis=1表示按行相加 , axis=0表示按列相加
    distances = sqlDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 距离最小的样本从小到大排序 返回数据索引值
    classCount = {}  # 定义一个字典
    for i in range(k):
        votellable = labels[sortedDistIndicies[i]]  # 获取排序后，前k个样本的label
        classCount[votellable] = classCount.get(votellable, 0) + 1  # 指定默认值为0
        # 根据字典的第二个值进行排序 从大到小 python2 API为 classCount.iteritems() python3为classCount.items()
    SortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return SortedClassCount[0][0]  # 取出第一个元素的key值


# 约会分类
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minValues = dataSet.min(0)  # min(0)返回该矩阵中每一列的最小值 min(1)返回该矩阵中每一行的最小值 结果依然是矩阵
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    # print (shape(dataSet))  # shape(dataSet) 返回矩阵的行数和列数（999，3）
    normDataSet = zeros(shape(dataSet))  # 创建一个元素是0 （999，3）的矩阵
    m = dataSet.shape[0]  # 矩阵的行数
    # print(tile(minValues, (m, 1)))  # 将 minValues 一行3列的矩阵 平铺为999 行和3列
    normDataSet = dataSet - tile(minValues, (m, 1))  # 当前矩阵的每个元素 减去 每列矩阵的最小值
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 归一化
    return normDataSet, ranges, minValues


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("/Users/yangshaojun/python_workspace/chapter02/data/datingTestSet2.txt")
    normDataSet, ranges, minValues = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        calssResult = classify0(normDataSet[i, :], normDataSet, datingLabels, 3)
        print("分类器的分类结果 :%d, 真实结果是: %d  " % (calssResult, datingLabels[i]))
        if (calssResult != datingLabels[i]): errorCount += 1.0
    print("错误率是: %f " % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['一点不喜欢', '有一点喜欢', '非常喜欢']  # 结果类型
    # python2 API 为raw_input、 python3 API为input
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("/Users/yangshaojun/python_workspace/chapter02/data/datingTestSet2.txt")
    normDataSet, ranges, minValues = autoNorm(datingDataMat)
    inArray = array([percentTats, ffMiles, iceCream])  # 输入的数据作为一个矩阵
    # 进行分类预测
    classifierResult = classify0(inArray - minValues / ranges, datingDataMat, datingLabels, 3)
    print("你很可能是这么一个人: %s" % (resultList[classifierResult - 1]))


def img2vector(filename):
    returnVect = zeros((1, 1024))  # 定义一个矩阵 1行 1024列 元素值为0
    file = open(filename)
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])  # 将32 行 32 列的矩阵 转为 1 行 1024 列矩阵
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 引入 os 模块的 listdir 显示指定文件夹下的所有的文件名称
    trainingFileList = listdir("/Users/yangshaojun/python_workspace/chapter02/trainingDigits")
    m = len(trainingFileList)  # 获取文件的个数
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('/Users/yangshaojun/python_workspace/chapter02/trainingDigits/%s' % fileNameStr)
    testFileList = listdir("/Users/yangshaojun/python_workspace/chapter02/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)  # 测试集 文件数目
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/Users/yangshaojun/python_workspace/chapter02/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器的分类结果 :%d, 真实结果是: %d  " % (classifierResult, classNumStr))

        if (classifierResult != classNumStr): errorCount += 1.0
    print("错误率是: %f " % (errorCount / float(mTest)))


if __name__ == '__main__':
    group, labels = createDataSet()  # 创建数据集
    ret = classify0([0, 0], group, labels, 3)  # 返回分类标签
    print(ret)
    datingDataMat, datingLabels = file2matrix("/Users/yangshaojun/python_workspace/chapter02/data/datingTestSet2.txt")
    # matplotlib 展示
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()
    # 数据归一化
    normDataSet, ranges, minValues = autoNorm(datingDataMat)
    # print(normDataSet)
    datingClassTest()
    classifyPerson()
    img2vector("/Users/yangshaojun/python_workspace/chapter02/trainingDigits/0_13.txt")
    handwritingClassTest()
