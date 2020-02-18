# -*- coding:UTF-8 -*-
from numpy import *
from time import sleep
import matplotlib.pyplot as plt

"""
函数说明:加载数据集

Parameters:
    fileName:文件名称
Returns:
    dataMat，labelMat
"""


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

"""
函数说明:展示数据

Parameters:
    dataMat, labelMat:
Returns:
"""
def showDataSet(dataMat, labelMat):
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)              #转换为numpy矩阵
    data_minus_np = array(data_minus)            #转换为numpy矩阵
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1]) #负样本散点图
    plt.show()


"""
函数说明:
Parameters:
    i是alpha的下标，
    m是所有的alpha的数目
Returns:
    调整 aj的大小；使其介于 L<= aj <=H 之间
"""

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


"""

函数说明:调整aj数据的大小
        调整大于H或小于L的值
        使得 L<= aj <=H
Parameters:
    aj、dataMat、labelMat 
Returns:
    调整 aj的大小；使其介于 L<= aj <=H 之间
"""


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
函数说明:简化版SMO算法

Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    无
"""

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #转换为numpy的mat存储
    dataMatrix = mat(dataMatIn) # 100 x 2 矩阵
    labelMat = mat(classLabels).transpose() # 将类别标签列表转为矩阵 然后进行矩阵的转置 得到100 x 1 的矩阵
    #初始化b参数，统计dataMatrix的维度
    b = 0; m,n = shape(dataMatrix)
    #初始化alpha参数，设为0
    alphas = mat(zeros((m,1))) # 创建一个 100 x 1 的矩阵 每个 alpha 对应着 每个样本
    #初始化迭代次数
    iter = 0
    #最多迭代matIter次
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            """
            # 数组：(点对点）对应位置元素相乘
            # 矩阵：对应位置元素相乘
            # multiply(alphas, labelMat) 返回 100 x 1 矩阵 
            # 操作符 .T 矩阵的转置  multiply(alphas, labelMat).T 结果 1 x 100 矩阵
            # dataMatrix*dataMatrix[j,:].T  操作： 100 x 2 外积  2 * 1 矩阵  结果是 100 x 1 矩阵
            """
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 步骤1：计算误差Ei
            Ei = fXi - float(labelMat[i])
            # 优化alpha，更设定一定的容错率。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i,m)
                # 步骤1：计算误差Ej
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print ("eta>=0"); continue
                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

"""
函数说明:分类结果可视化
Parameters:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线解决
Returns:
    无
"""
def showClassifer(dataMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)              #转换为numpy矩阵
    data_minus_np = array(data_minus)            #转换为numpy矩阵
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


"""
函数说明:计算w
Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
    alphas - alphas值
Returns:
    无
"""
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = array(alphas), array(dataMat), array(labelMat)
    # labelMat.reshape(1, -1).T 得到 100 x 1
    # tile(labelMat.reshape(1, -1).T, (1, 2)) 100 x 2
    # dataMat 100 x 2 矩阵
    # alphas 100 x 1
    #  w 2 x 1 矩阵
    w = dot((tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
    # 加载数据集
    dataMat, labelMat = loadDataSet('testSet.txt')
    # 图示展示
    showDataSet(dataMat,labelMat)
    b,alphas=smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    print(w)
    showClassifer(dataMat, w, b)

