from numpy import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet(fileName):
    """
    加载数据集
    :param fileName:  文件名称
    :return: 返回特征数据和标签数据
    """
    numFeat = len(open(fileName).readline().split("\t")) - 1  # 返回特征值
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegress(xArr, yArr):
    """
    标准的线性回归方法
    :param xArr: 特征数据
    :param yArr: 标签数据
    :return: 返回权重系数
    """
    xMat = mat(xArr)  # 特征列向量
    yMat = mat(yArr).T  # 标签列向量
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)  # 求解w值
    return ws


def showGraph(xArr, yArr, ws, k):
    """
    绘制原始数据然后找到最佳拟合直线
    :param xArr:
    :param yArr:
    :param ws:
    :return:
    """
    xMat = mat(xArr)  # 特征列向量
    yMat = mat(yArr).T  # 标签列向量
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0])  # 绘制原始数据
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws  # 根据回归函数 预测的标签值
    ax.plot(xCopy[:, 1], yHat)  # 绘制拟合直线
    plt.show()
    relation = corrcoef(yHat.T, yMat.T)  # 计算相关系数
    print(relation)


def lwlr(testPoint, xArr, yArr, k=1):
    """
    局部线性回归
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建一个 m x m 列的矩阵 矩阵元素为0 对角线值为1
    for j in range(m):
        diffMat = testPoint - xMat[j, :]  # xMat[j,:] 表示第j个样本 预测点和所有样本点的距离
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))  # 高斯核 给每个样本元素加权重 和预测点越近的点权重越大
    xTx = xMat.T * (weights * xMat)  # (weights * xMat) 给每个特征值加一个权重
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(xArrTest, xArr, yArr, k=0.01):
    """
    获取数据集中所有点的估计
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xArrTest)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(xArrTest[i], xArr, yArr, k)
    srtInd = xMat[:, 1].argsort(0)  # 将元素从小到大排列 然后返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat[:, 0].flatten().A[0])  # 绘制原始数据
    ax.plot(xSort[:, 1], yHat[srtInd])  # 绘制拟合直线
    plt.show()
    return yHat


def ridgeRegres(xMat, yMat, lam=0.2):
    """
    函数说明:岭回归
    Parameters:
        xMat - x数据集
        yMat - y数据集
        lam - 缩减系数
    Returns:
        ws - 回归系数
    """
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    """
    函数说明:岭回归测试
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        wMat - 回归系数矩阵
    """
    xMat = mat(xArr)  # 4177 x 8 矩阵
    yMat = mat(yArr).T  # 4177 x 1 矩阵
    # 数据标准化
    yMean = mean(yMat, axis=0)  # 求均值 mean(axis=0) 其中 axis=0 求每列的均值 axis=1求没行的均值
    yMat = yMat - yMean  # 数据减去均值
    xMeans = mean(xMat, axis=0)  # 求样本数据每个特征的均值
    xVar = var(xMat, axis=0)  # 数据样本中每个特征，求方差
    xMat = (xMat - xMeans) / xVar  # 数据减去均值除以方差实现标准化
    numTestPts = 30  # 30个不同的lambda测试
    wMat = zeros((numTestPts, shape(xMat)[1]))  # 初始回归系数矩阵 30 x 8 矩阵
    for i in range(numTestPts):  # 改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, exp(i - 10))  # lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T  # 计算回归系数矩阵
    return wMat


def plotwMat():
    """
    函数说明:绘制岭回归系数矩阵
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-11-20
    """
    font = FontProperties(fname=r"./font/simsun.ttc", size=14)
    abX, abY = loadDataSet('./data/abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def regularize(xMat, yMat):
    """
    函数说明:数据标准化
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集
    """
    inxMat = xMat.copy()  # 数据拷贝
    inyMat = yMat.copy()
    yMean = mean(yMat, 0)  # 行与行操作，求均值
    inyMat = yMat - yMean  # 数据减去均值
    inMeans = mean(inxMat, 0)  # 行与行操作，求均值
    inVar = var(inxMat, 0)  # 行与行操作，求方差
    inxMat = (inxMat - inMeans) / inVar  # 数据减去均值除以方差实现标准化
    return inxMat, inyMat


def rssError(yArr, yHatArr):
    """
    函数说明:计算平方误差
    Parameters:
        yArr - 预测值
        yHatArr - 真实值
    Returns:
    """
    return ((yArr - yHatArr) ** 2).sum()


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    函数说明:前向逐步线性回归
    Parameters:
        xArr - x输入数据
        yArr - y预测数据
        eps - 每次迭代需要调整的步长
        numIt - 迭代次数
    Returns:
        returnMat - numIt次迭代的回归系数矩阵
    """

    xMat = mat(xArr)
    yMat = mat(yArr).T  # 数据集
    xMat, yMat = regularize(xMat, yMat)  # 数据标准化
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  # 初始化numIt次迭代的回归系数矩阵
    ws = zeros((n, 1))  # 初始化回归系数矩阵
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):  # 迭代numIt次
        lowestError = float('inf')  # 正无穷
        for j in range(n):  # 遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign  # 微调回归系数
                yTest = xMat * wsTest  # 计算预测值
                rssE = rssError(yMat.A, yTest.A)  # 计算平方误差
                if rssE < lowestError:  # 如果误差更小，则更新当前的最佳回归系数
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T  # 记录numIt次迭代的回归系数矩阵
    return returnMat


def plotstageWiseMat():
    """
    函数说明:绘制岭回归系数矩阵
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-11-20
    """
    font = FontProperties(fname=r"./font/simsun.ttc", size=14)
    xArr, yArr = loadDataSet('./data/abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    # xArr, yArr = loadDataSet("./data/ex0.txt")
    # ws = standRegress(xArr, yArr)
    # showGraph(xArr, yArr, ws, "")
    # ret = lwlr(xArr[0], xArr, yArr, k=0.03)
    # retHat = lwlrTest(xArr, xArr, yArr, k=0.01)

    xArr, yArr = loadDataSet("./data/abalone.txt")
    # 岭回归
    # plotwMat()
    # 向前逐步线性回归
    plotstageWiseMat()
