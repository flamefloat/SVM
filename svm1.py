import math
import numpy as np
from numpy import * 
import scipy.io as scio
import matplotlib.pyplot as plt

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
        self.data_num = data.shape[0] # 训练集长度
        self.data_width = data.shape[1] #实例维度
        self.a = np.zeros((self.data_num, 1))
        self.b = 0
        self.E = np.zeros((self.data_num,2))
        self.K = np.zeros((self.data_num,self.data_num))
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

def findJ(svm, k, Ei): #根据a1,寻找a2
    Ej = 0
    maxE = 0
    pos_j = 0
    """
    if Ei >=0:
        pos_j = np.argmin(E)  
    else:
        pos_j = np.argmax(E)
    return pos_j
    """
    validEcacheList=[]
    for i in range(svm.data_num):
        if svm.E[i,0] > 0:
            validEcacheList.append(i)   
    if len(validEcacheList) > 1:
        for i in validEcacheList:
            if k == i:
                continue
            tempE = calcError(svm,i)
            if maxE < abs(Ei - tempE):
                maxE = abs(Ei - tempE)
                pos_j = i
                Ej = tempE
        return pos_j, Ej
    else:
        l = list(range(svm.data_num))
        # 排除掉已选的 i
        seq = l[:i] + l[i + 1:]
        j = random.choice(seq)
        Ej = calcError(svm, j)
    return j, Ej






def updateParam(svm, i):
    Ei = calcError(svm, i)
    if ((svm.label[i]*Ei < -svm.toler) and (svm.a[i] < svm.c))\
        or ((svm.label[i]*Ei > svm.toler) and (svm.a[i] > 0)): # 违反KKT条件的a1
        j, Ej = findJ(svm, i, Ei)
        old_a1 = svm.a[i]
        old_a2 = svm.a[j]
        eta = svm.K[i,i] + svm.K[j,j] - 2.0 * svm.K[i,j]
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
        svm.E[i,:] = [1, calcError(svm, i)]
        svm.E[j,:] = [1, calcError(svm, j)]
        return 1
    else:
        return 0

def smoTrain(svm, maxIter):
    iterCount = 0
    #svm.E = -svm.label #初始化误差
    entireSet = True
    #alphaPairsChanged = 0
    while (iterCount < maxIter):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(svm.data_num):
                alphaPairsChanged += updateParam(svm, i)
            #loss = lossCount(svm)
            print('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
        
        else:  # 对边界上的alpha遍历(即约束在0<a1<C内的样本点)
            nonBoundIs = []
            for i in range(svm.data_num):
                if (0 < svm.a[i] and svm.a[i] < svm.c):
                    nonBoundIs.append(i)
            for i in nonBoundIs:
                alphaPairsChanged += updateParam(svm, i)
            print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
        iterCount += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        
        print('*********E:',sum(abs(svm.E)))
    print(sum(svm.E[:,0]))
    return svm

def calcWB(svm):
    w = np.zeros((svm.data_width))
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

if __name__ == '__main__':
    D1 = scio.loadmat('C:\\Users\\MH\\Desktop\\MyCode\\MLalgorithm\\SVM\\Data1.mat')
    data = D1['data']
    #print(data[1:5,:])
    mySVM = SVM(data[0:70,0:2], data[0:70,2], 1, 1, 0.001, 'linear')
    mySVM = smoTrain(mySVM, 10)
    print(mySVM.a)
    w, b = calcWB(mySVM)
    for i in range(70):
        if mySVM.a[i] > 0:
            plt.plot(data[i, 0], data[i, 1], 'oy')
        else:
            if data[i,2] == 1:
                plt.plot(data[i, 0], data[i, 1], 'or')
            if data[i,2] == -1:
                plt.plot(data[i, 0], data[i, 1], 'ob')
    x = np.arange(0,1,0.05)
    y = -(w[0]/w[1]) * x - b/w[1]
    plt.plot(x,y,'-g')
    plt.legend(('SV', 'y = 1', 'y = -1','f(x)'),
           loc='upper right')
    plt.title('C = 10')
    plt.show()

    


