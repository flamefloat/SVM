import math
import numpy as np
from numpy import * 
import scipy.io as scio
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        linArr = line.strip().split('\t')
        dataMat.append([float(linArr[0]), float(linArr[1])])
        labelMat.append(float(linArr[2]))
    return dataMat, labelMat

class SVM():
    """
    data:训练数据集，NxM矩阵，实列数：N,特征向量维度：M
    label:标签，1xN的矩阵，取值为-1，1
    c:惩罚系数
    sigma:RBF核函数参数
    """
    def __init__(self, data, label, c, sigma, toler, kernelOpt):
        self.data = data
        self.label = label
        self.c = c
        self.sigma = sigma
        self.toler = toler #KKT条件精度
        #self.maxIter = maxIter
        self.kernelOpt = kernelOpt
        self.data_num = shape(data)[0] # 训练集长度
        #self.data_width = data.shape[1] #实例维度
        self.a = mat(zeros((self.data_num, 1)))
        self.b = 0
        self.errorCache = mat(zeros((self.data_num, 2)))
        #self.E = np.zeros(self.data_num)
        self.K = mat(zeros((self.data_num,self.data_num)))
        for i in range(self.data_num):
            for j in range(self.data_num):
                self.K[i,j] = kernel(self.data[i,:], self.data[j,:], self.sigma, self.kernelOpt)

def kernel(x1, x2, sigma, opt): # 默认RBF核函数
    if opt == 'linear':
        return sum(x1*x2)
    else:
        return math.exp(-sum((x1-x2)**2)/(2*sigma**2))

def g(svm, xk): # 函数g(x),xk:实例的下标
        temp = 0
        for i in range(svm.data_num):
            temp += svm.a[i]*svm.label[i]*svm.K[i,xk]
        temp += svm.b
        return temp

def calcError(svm, xk): #计算样本K的预测误差
    #tempE = sum(svm.label*svm.a*svm.K[:,xk]) + svm.b
    tempE = float(sum(svm.a*svm.label* svm.K[:, xk]) + svm.b)
    EK = tempE - svm.label[xk]
    return EK

def updateError(svm, k):
    """
    第k个样本的误差存入缓存矩阵，再选择第二个alpha值用到
    :param svm:
    :param k: 样本索引
    :return:
    """
    Ek = calcError(svm, k)
    svm.errorCache[k] = [1, Ek]

"""
def findJ(svm, i, E): #根据a1,寻找a2
    Ei = E[i]
    if Ei >=0:
        pos_j = np.argmin(E)  
    else:
        pos_j = np.argmax(E)
    return pos_j
"""
def selectJ(svm, i, Ei):
    """
    寻找第二个待优化的alpha,并具有最大步长
    :param i: 第一个alpha值的下标
    :param svm:
    :param Ei:第一个alpha值对应的Ei
    :return:
    """
    maxK = 0
    maxStep = 0
    Ej = 0
    validEcacheList = nonzero(svm.errorCache[:, 0].A)[0]  # 从误差缓存矩阵中 得到记录所有样本有效标志位的列表(注：存的是索引)
    if (len(validEcacheList)) > 1:  # 选择具有最大步长的 j
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcError(svm, k)
            step = abs(Ei - Ek)
            if (step > maxStep):  # 选择 Ej 与 Ei 相差最大的那个 j，即步长最大
                maxK = k
                maxStep = step
                Ej = Ek
        return maxK, Ej
    else:  # 第一次循环采用随机选择法
        l = list(range(svm.data_num))
        # 排除掉已选的 i
        seq = l[:i] + l[i + 1:]
        j = random.choice(seq)
        Ej = calcError(svm, j)
    return j, Ej


def updateParam(svm, i):
    Ei = calcError(svm, i)
    if ((svm.label[i]*Ei < -svm.toler) and (svm.a[i] < svm.c))\
        or ((svm.label[i]*Ei > svm.toler) and (svm.a[i] > 0)):
        j, Ej = selectJ(svm, i, Ei)
        #E[j] = Ej
        old_a1 = svm.a[i]
        old_a2 = svm.a[j]
        eta = svm.K[i,i] + svm.K[j,j] - 2.0 *svm.K[i,j]
        temp_a2 = old_a2 + svm.label[j] * (Ei - Ej) / eta
        if svm.label[i] == svm.label[j]:
            L = max(0, old_a2 + old_a1 - svm.c)
            H = min(svm.c, old_a2 + old_a1)
        else:
            L = max(0, old_a2 - old_a1)
            H = min(svm.c, svm.c + old_a2 - old_a1)
        if temp_a2 > H:
            svm.a[j] = H
        elif temp_a2 < L:
            svm.a[j] = L
        else:
            svm.a[j] = temp_a2
        svm.a[i] = old_a1 + svm.label[i] * svm.label[j] * (old_a2 - svm.a[j])
        new_b1 = -Ei\
                - svm.label[i]*svm.K[i,i]*(svm.a[i]-old_a1)\
                - svm.label[j]*svm.K[j,i]*(svm.a[j]-old_a2)\
                + svm.b
        new_b2 = -Ej\
                - svm.label[i]*svm.K[i,j]*(svm.a[i]-old_a1)\
                - svm.label[j]*svm.K[j,j]*(svm.a[j]-old_a2)\
                + svm.b
        if (svm.a[i] > 0) and (svm.a[i] < svm.c):
            svm.b = new_b1
        elif (svm.a[j] > 0) and (svm.a[j] < svm.c):
            svm.b = new_b2
        else:
            svm.b = (new_b1 + new_b2) / 2
        #E[i] = calcError(svm, i)
        #E[j] = calcError(svm, j)
        updateError(svm, j)
        updateError(svm, i)
        return 1
    else:
        return 0

def smoTrain(svm, maxIter):
    iterCount = 0
    #E = -svm.label #初始化误差
    entireSet = True
    #alphaPairsChanged = 0
    while (iterCount < maxIter):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(svm.data_num):
                alphaPairsChanged += updateParam(svm, i)
            #loss = lossCount(svm)
            print('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
        else:  # 对非边界上的alpha遍历(即约束在0<alpha<C内的样本点)
            nonBoundIs = np.nonzero((svm.a.A > 0) * (svm.a.A < svm.c))[0]
            for i in nonBoundIs:
                alphaPairsChanged += updateParam(svm, i)
            print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
        iterCount += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return svm

def calcWB(svm):
    w = mat(zeros((1,svm.data_width)))
    temp_B = 0
    pos = 0
    for i in range(svm.data_num):
        w += svm.a[i]*svm.label[i]*svm.data[i,:]
        if svm.a[i] > 0 and svm.a[i] < svm.c:
            pos = i
    print('w：',w)
    for i in range(svm.data_num):
        temp_B += svm.a[i]*svm.label[i]*svm.K[i,pos]
    B = svm.label[pos] - temp_B
    return w, B

def lossCount(svm):
        loss = 0
        for i in range(svm.data_num):
            for j in range(svm.data_num):
                loss += svm.a[i]*svm.a[j]*svm.label[i]*svm.label[j]\
                       *svm.K[i,j]
        loss = 0.5*loss-sum(svm.a)
        return loss

def plotSVM():
    dataMat, labelMat = loadDataSet('testSet.txt')
    svm = SVM(dataMat, labelMat, 1, 1, 0.001, 'linear')
    svm = smoTrain(svm, 50)

    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataMat, labelMat):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label, pts in classified_pts.items():
        pts = array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

    w = calcWs(svm.alphas, dataMat, labelMat)

    x1 = min(array(dataMat)[:, 0])
    x2 = max(array(dataMat)[:, 0])

    a1, a2 = w
    y1, y2 = (-float(svm.b) - a1 * x1) / a2, (-float(svm.b) - a1 * x2) / a2
    ax.plot([x1, x2], [y1, y2])
    plt.show()

if __name__ == '__main__':
    plotSVM()