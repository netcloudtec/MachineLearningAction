# -*- coding: UTF-8 -*-
import numpy as np
import operator

"""
函数说明:创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
Modify:
    2020-06-24
"""

def createDataSet():
    # 创建四组二维特征
    data = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])  # 使用 numpy的array方法创建一个矩阵
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return data, labels


"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
Modify:
    2020-06-24
"""

def classify(inX, dataSet, labels, k):
    # numpy函数的 shape[m,n]返回dataSet的 行x列 数目
    dataSetSize = dataSet.shape[0]
    # numpy的tile方法进行数据的重复 具体看API
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    data, labels = createDataSet()
    # 待预测的样本
    test = [101, 20]
    # kNN进行分类
    classResult = classify(test, data, labels, 3)
    # 输出分类结果
    print(classResult)
