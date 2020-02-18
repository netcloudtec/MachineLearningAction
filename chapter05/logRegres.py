# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

"""
函数说明:加载数据
Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
Author:
    yangshaojun
Modify:
    2020-02-01
"""


# 加载数据 特征变为3维 吴恩达通常令第一个特征为1 即w0
def loadDataSet():
    dataMat = []  # 此列表存储特征
    labelMat = []  # 此列表存标签
    fr = open('testSet.txt')  # 打开文件
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split('\t')  # 去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 每个样本特征作为列表存储
        labelMat.append(int(lineArr[2]))  # 添加到标签列表中
    fr.close()  # 关闭文件
    return dataMat, labelMat  # 返回


"""
函数说明:绘制数据集
Parameters:
    无
Returns:
    无
Author:
    yangshaojun
Modify:
    2020-02-01
"""
def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataArr)[0]  # 数据样本的数目
    xcode1 = []; ycode1 = []  #正样本
    xcode2 = []; ycode2 = []  #负样本
    for i in range(n):        #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcode1.append(dataArr[i, 1]); ycode1.append(dataArr[i, 2]) #1为正样本
        else:
            xcode2.append(dataArr[i, 1]); ycode2.append(dataArr[i, 2]) #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111) #添加subplot
    ax.scatter(xcode1, ycode1, s=20, c='red', marker='s', alpha=0.5) #绘制正样本
    ax.scatter(xcode2, ycode2, s=20, c='green', alpha=0.5) #绘制负样本
    plt.title('DataSet') #绘制title
    plt.xlabel('x')  #绘制label
    plt.ylabel('y')
    plt.show()  #显示

"""
函数说明:sigmoid函数
Parameters:
    inX - 数据
Returns:
    sigmoid函数
Author:
    yangshaojun     
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法

Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
    [[ 4.12414349]
    [ 0.48007329]
    [-0.6168482 ]]
Author:
    yangshaojun
Modify:
    2020-02-01
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 列表转换成numpy的mat 100行3列
    labelMat = np.mat(classLabels).transpose() #转换成numpy的mat,并进行转置 100行1列
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001  #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))  # 创建一个3行1列的矩阵 矩阵元素为1
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  #梯度上升矢量化公式
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles,n)
    return weights.getA()
    # return weights.getA(),weights_array  # 将矩阵转换为数组，返回权重数组


"""
函数说明:绘制数据集

Parameters:
    weights - 权重参数数组
Returns:
    无
Author:
    yangshaojun
Modify:
    2020-02-01
"""

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  #转换成numpy的array数组
    n = np.shape(dataArr)[0]  # 返回样本数
    xcode1 = []; ycode1 = []  #正样本
    xcode2 = []; ycode2 = []  #负样本
    for i in range(n):        #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcode1.append(dataArr[i, 1]);  ycode1.append(dataArr[i, 2]) #1为正样本
        else:
            xcode2.append(dataArr[i, 1]);  ycode2.append(dataArr[i, 2]) #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           #添加subplot
    ax.scatter(xcode1, ycode1, s=30, c='red', marker='s')               #绘制正样本
    ax.scatter(xcode2, ycode2, s=30, c='green')                         #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # 这里的y的值即x2
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    dataArr=np.array(dataMatrix)
    m, n = np.shape(dataArr)
    alpha = 0.01
    weights = np.ones(n)  # initialize to all ones
    for i in range(m):
        # h,error全是数值，没有矩阵转换过程。
        h = sigmoid(sum(dataArr[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataArr[i]
    return weights


"""
函数说明:改进的随机梯度上升算法

Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
    weights_array - 每次更新的回归系数
Author:
    yangshaojun
Modify:
    2020-02-01
"""
# 随机梯度上升算法改进
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataArr=np.array(dataMatrix)
    m, n = np.shape(dataArr)
    weights = np.ones(n)  #参数初始化
    weights_array = np.array([]) #存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机抽取样本来更改取样样本，减少周期性的波动
            h = sigmoid(sum(dataArr[randIndex] * weights))      # 选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                  # 计算误差
            weights = weights + alpha * error * dataArr[randIndex] # 更新回归系数
            weights_array = np.append(weights_array,weights,axis=0) # 添加回归系数到数组中
            del (dataIndex[randIndex])                             # 删除已经使用的样本
    weights_array = weights_array.reshape(numIter*m,n)             # 改变维度
    return weights
    # return weights,weights_array


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
Returns:
    无
Author:
    yangshaojun
Modify:
    2020-02-01
"""
def plotWeights(weights_array1,weights_array2):
    #设置汉字格式
    font = FontProperties(fname=r"/Users/yangshaojun/python_workspace/chapter05/font/simsun.ttc", size=14)  # 设置中文字体

    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print(weights)
    plotBestFit(weights)
    weights = stocGradAscent0(dataArr, labelMat)
    plotBestFit(weights)
    weights = stocGradAscent1(dataArr, labelMat)
    plotBestFit(weights)
    multiTest()
    plotDataSet()

    # dataMat, labelMat = loadDataSet()
    # weights1, weights_array1 = stocGradAscent1(dataMat, labelMat)
    #
    # weights2, weights_array2 = gradAscent(dataMat, labelMat)
    # plotWeights(weights_array1, weights_array2)
