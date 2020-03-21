# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

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
        xcord.append(dataMat[i][0]);
        ycord.append(dataMat[i][1])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def binSplitDataSet(dataSet, feature, value):
    """
    函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def linearSolve(dataSet):  # helper function used in two places
    """
    将数据格式化为目标变量Y和自变量X
    :param dataSet:
    :return:
    """
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)));
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1];
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # 当数据不在需要切分的时候，生成叶子节点模型
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):  # 计算误差
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


def chooseBestSplit(dataSet, leafType=modelLeaf, errType=modelErr, ops=(1, 10)):
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
    import types
    # tolS允许的误差下降值,tolN切分的最少样本数
    tolS = ops[0];
    tolN = ops[1]
    # 如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 统计数据集合的行m和列n
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    # 遍历所有特征列
    for featIndex in range(n - 1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN,则退出
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            # 计算误差估计
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 10)):
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
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # r如果没有特征,则返回特征值
    if feat == None: return val
    # 回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')

# 回归树测试案例
# 为了和 modelTreeEval() 保持一致，保留两个输入参数
# 模型效果计较
# 线性叶子节点 预测计算函数 直接返回 树叶子节点 值
def regTreeEval(model, inDat):
    return float(model)

# 模型树测试案例
# 对输入数据进行格式化处理，在原数据矩阵上增加第1列，元素的值都是1，
# 也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)

# 计算预测的结果
# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
# modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
# 调用modelEval()函数，该函数的默认值为regTreeEval()
def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
        Desc:
            对特定模型的树进行预测，可以是 回归树 也可以是 模型树
        Args:
            tree -- 已经训练好的树的模型
            inData -- 输入的测试数据
            modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
        Returns:
            返回预测值
        """
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

# 得到预测值
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

#使用Tkinter创建GUI
def guiTest():
    print("GUI")
    root=tk.Tk()
    myLabel=tk.Label(root,text="hello World")
    myLabel.grid()
    root.mainloop()

if __name__ == '__main__':
    train_filename = './data/exp2.txt'
    plotDataSet(train_filename)
    train_Data = loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    myTree = createTree(train_Mat,modelLeaf, modelErr, ops=(1, 10))
    print(myTree)

    train_Data = loadDataSet('./data/bikeSpeedVsIq_train.txt')
    train_Mat = np.mat(train_Data)
    test_Data = loadDataSet('./data/bikeSpeedVsIq_test.txt')
    test_Mat = np.mat(test_Data)
    # 回归树相关系数
    tree = createTree(train_Mat, ops=(1, 20))
    YHat = createForeCast(tree, test_Mat[:, 0], regTreeEval)
    corr = np.corrcoef(YHat, test_Mat[:, 1], rowvar=0)[0][1]
    print('普通回归树 预测结果的相关系数R2: %f'%(corr))

    # 模型树相关系数
    tree = createTree(train_Mat, modelLeaf, modelErr, ops=(1, 20))
    YHat = createForeCast(tree, test_Mat[:, 0], modelTreeEval)
    corr = np.corrcoef(YHat, test_Mat[:, 1], rowvar=0)[0][1]
    print('模型回归树 预测结果的相关系数R2: %f'%(corr))

    # 线性模型相关系数
    ws, X, Y = linearSolve(train_Mat)
    for i in range(np.shape(test_Mat)[0]):
        YHat[i] = test_Mat[i, 0] * ws[1, 0] + ws[0, 0]
    corr = np.corrcoef(YHat, test_Mat[:, 1], rowvar=0)[0, 1]
    print('线性回归模型 预测结果的相关系数R2: %f'%(corr))

    guiTest()
