# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC

def img2Vector(filename):
    """
    将32x32的二进制图像转换为1x1024向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
    """
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1x1024向量
    return returnVect


def handwritingClassTest():
    # 存储训练集的标记
    hwLabels = []
    trainDataDir = '/Users/yangshaojun/python_workspace/chapter02/trainingDigits'
    # 返回此目录下的所有文件名称
    trainingFileList = listdir(trainDataDir)
    # 统计文件的个数
    m = len(trainingFileList)  # 1934个文件
    # 定义矩阵的大小
    traingMat = np.zeros((m, 1024))
    # 遍历每个文件 将其转为矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 根据文件名称获取类别
        classNumber = int(fileNameStr.split('_')[0])
        # 将每个文件的类别加到hwlabels
        hwLabels.append(classNumber)
        traingMat[i, :] = img2Vector(trainDataDir + '/' + fileNameStr)
    # 这里是有的核函数是
    clf = SVC(C=200, kernel='rbf')
    clf.fit(traingMat, hwLabels)
    testDataDir = '/Users/yangshaojun/python_workspace/chapter02/testDigits'
    testFileList = listdir(testDataDir)
    errorCount = 0.0
    mTest = len(testFileList)
    testMat = np.zeros((mTest, 1024))
    testLabels = []
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2Vector(testDataDir + '/' + fileNameStr)
        classifierResult = clf.predict(vectorUnderTest)
        testMat[i, :] = vectorUnderTest
        testLabels.append(classNumber)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))



if __name__ == '__main__':
    handwritingClassTest()
