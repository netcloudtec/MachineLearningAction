# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 转化为float类型
        dataMat.append(fltLine)
    return dataMat


def plotDataSet(filename):
    """
    函数说明:绘制数据集
    Parameters:
        filename - 文件名
    Returns:
        无
    """
    dataMat = loadDataSet(filename)  # 加载数据集
    n = len(dataMat)  # 数据个数
    xcord = [];
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(dataMat[i][1]);
        ycord.append(dataMat[i][2])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def regLeaf(dataSet):
    """
    函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
    """
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    """
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差
    """
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def binSplitDataSet(dataSet, feature, value):
    """
    :param dataSet: 训练集
    :param feature: 特征索引 1
    :param value: 特征值  0.5
    :return: mat0, mat1 切分的数据集
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]  # [:] 列表切片 索引从0开始 包括开始不包括结束 [:1] 表示训练集的第一列；[1,:]表示训练集的第2行
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]  # np.nonzero：返回数组中不为0的元素的下标。数组中元素可为布尔、整型和浮点型，返回值为元祖
    return mat0, mat1


def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    """
      函数说明:找到数据的最佳二元切分方式函数
      Parameters:
          dataSet - 数据集合
          leafType - 生成叶结点
          regErr - 误差估计函数
          ops - 用户定义的参数构成的元组
      Returns:
          bestIndex - 最佳切分特征
          bestValue - 最佳特征值
      """
    # tolS允许的误差下降值,tolN切分的最少样本数
    tolS=ops[0]
    tolN=ops[1]
    # 如果当前样本所有标签值相等,则退出。(根据set的特性)
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    # 统计数据集合的行m和列n
    m,n=np.shape(dataSet) # 200 x 2
    #默认最后一个特征为最佳切分特征,计算其误差估计
    S=errType(dataSet)
    #分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS=float('inf')# 正无穷
    bestInfo=0
    bestValue=0
    #遍历所有特征列
    for featIndex in range(n-1):
        #遍历所有特征值
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            #根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #如果数据少于tolN,则退出
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            #计算误差估计
            newS = errType(mat0) + errType(mat1)
            #如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
   #如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    #返回最佳切分特征和特征值
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    """
    函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树
    """
    #选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    #r如果没有特征,则返回特征值
    if feat == None: return val
    #回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    #创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


if __name__ == '__main__':
    filename = './data/ex0.txt'
    plotDataSet(filename)
    myDat = loadDataSet(filename)
    myMat = np.mat(myDat)
    print(createTree(myMat))


