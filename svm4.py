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
"""
def calcError(svm, xk): #计算样本K的预测误差
    #tempE = sum(svm.label*svm.a*svm.K[:,xk]) + svm.b
    tempE = float(sum(svm.a*svm.label* svm.K[:, xk]) + svm.b)
    EK = tempE - svm.label[xk]
    return EK
"""
def calcError(svm, alpha_k):
    output_k = 0
    #output_k = float(sum( (svm.a*svm.label) * svm.K[:, alpha_k] )+ svm.b)
    for i in range(svm.data_num):
        output_k += svm.a[i]*svm.label[i]*svm.K[i,alpha_k]
    output_k += svm.b
    error_k = output_k - float(svm.label[alpha_k])
    return error_k

def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.E[alpha_k] = [1, error]

"""
def findJ(svm, k, Ei): #根据a1,寻找a2
    svm.E[k] = [1, Ei]
    Ej = 0
    maxE = 0
    pos_j = 0
    
    if Ei >=0:
        pos_j = np.argmin(E)  
    else:
        pos_j = np.argmax(E)
    return pos_j
    
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
    else:
        l = list(range(svm.data_num))
        # 排除掉已选的 i
        seq = l[:i] + l[i + 1:]
        pos_j = random.choice(seq)
        Ej = calcError(svm, pos_j)
    return pos_j, Ej
"""
def findJ(svm, alpha_i, error_i):
    svm.E[alpha_i] = [1, error_i] # mark as valid(has been optimized)
    candidateAlphaList = nonzero(svm.E[:, 0])[0] # mat.A return array
    maxStep = 0; alpha_j = 0; error_j = 0

    # find the alpha with max iterative step
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i: 
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    # if came in this loop first time, we select alpha j randomly
    else:           
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.data_num))
        error_j = calcError(svm, alpha_j)
    
    return alpha_j, error_j




"""
def updateParam(svm, i):
    Ei = calcError(svm, i)
    if ((svm.label[i]*Ei < -svm.toler) and (svm.a[i] < svm.c))\
        or ((svm.label[i]*Ei > svm.toler) and (svm.a[i] > 0)): # 违反KKT条件的a1
        j, Ej = findJ(svm, i, Ei)
        old_a1 = svm.a[i].copy()
        old_a2 = svm.a[j].copy()
        eta = svm.K[i,i] + svm.K[j,j] - 2.0 * svm.K[i,j]
        if eta < 0:
            return 0
        temp_a2 = old_a2 + svm.label[j] * (Ei - Ej) / eta
        if svm.label[i] == svm.label[j]:
            L = max(0, old_a2 + old_a1 - svm.c)
            H = min(svm.c, old_a2 + old_a1)
        else:
            L = max(0, old_a2 - old_a1)
            H = min(svm.c, svm.c + old_a2 - old_a1)
        if L == H:
            return 0
        if temp_a2 > H:
            svm.a[j] = H
        elif temp_a2 < L:
            svm.a[j] = L
        else:
            svm.a[j] = temp_a2

        if abs(old_a2 - svm.a[j]) < 0.00001:
            updateError(svm, j)
            return 0

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
        #svm.E[i,:] = [1, calcError(svm, i)]
        #svm.E[j,:] = [1, calcError(svm, j)]
        updateError(svm, j)
        updateError(svm, i)
        print('svm.a[i]',i,svm.a[i],'svm.a[j]',j,svm.a[j])
        return 1
    else:
        return 0
"""
def updateParam(svm, alpha_i):
    error_i = calcError(svm, alpha_i)

    ### check and pick up the alpha who violates the KKT condition
    ## satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    if (svm.label[alpha_i] * error_i < -svm.toler) and (svm.a[alpha_i] < svm.c) or\
        (svm.label[alpha_i] * error_i > svm.toler) and (svm.a[alpha_i] > 0):

        # step 1: select alpha j
        alpha_j, error_j = findJ(svm, alpha_i, error_i)
        alpha_i_old = svm.a[alpha_i].copy()
        alpha_j_old = svm.a[alpha_j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if svm.label[alpha_i] != svm.label[alpha_j]:
            L = max(0, svm.a[alpha_j] - svm.a[alpha_i])
            H = min(svm.c, svm.c + svm.a[alpha_j] - svm.a[alpha_i])
        else:
            L = max(0, svm.a[alpha_j] + svm.a[alpha_i] - svm.c)
            H = min(svm.c, svm.a[alpha_j] + svm.a[alpha_i])
        if L == H:
            return 0

        # step 3: calculate eta (the similarity of sample i and j)
        eta = 2.0 * svm.K[alpha_i, alpha_j] - svm.K[alpha_i, alpha_i] \
                  - svm.K[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # step 4: update alpha j
        svm.a[alpha_j] -= svm.label[alpha_j] * (error_i - error_j) / eta

        # step 5: clip alpha j
        if svm.a[alpha_j] > H:
            svm.a[alpha_j] = H
        if svm.a[alpha_j] < L:
            svm.a[alpha_j] = L

        # step 6: if alpha j not moving enough, just return        
        if abs(alpha_j_old - svm.a[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            return 0

        # step 7: update alpha i after optimizing aipha j
        svm.a[alpha_i] += svm.label[alpha_i] * svm.label[alpha_j] \
                                * (alpha_j_old - svm.a[alpha_j])

        # step 8: update threshold b
        b1 = svm.b - error_i - svm.label[alpha_i] * (svm.a[alpha_i] - alpha_i_old) \
                                                    * svm.K[alpha_i, alpha_i] \
                             - svm.label[alpha_j] * (svm.a[alpha_j] - alpha_j_old) \
                                                    * svm.K[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.label[alpha_i] * (svm.a[alpha_i] - alpha_i_old) \
                                                    * svm.K[alpha_i, alpha_j] \
                             - svm.label[alpha_j] * (svm.a[alpha_j] - alpha_j_old) \
                                                    * svm.K[alpha_j, alpha_j]
        if (0 < svm.a[alpha_i]) and (svm.a[alpha_i] < svm.c):
            svm.b = b1
        elif (0 < svm.a[alpha_j]) and (svm.a[alpha_j] < svm.c):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # step 9: update error cache for alpha i, j after optimize alpha i, j and b
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0
"""
def smoTrain(svm, maxIter):
    iterCount = 0
    alphaPairsChanged = 0
    #svm.E = -svm.label #初始化误差
    entireSet = True
    #alphaPairsChanged = 0
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(svm.data_num):
                alphaPairsChanged += updateParam(svm, i)
                #print('svm.a[i]',svm.a[i],'svm.a[j]',svm.a[j])
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
"""
def smoTrain(svm, maxIter):
    # calculate training time
    #startTime = time.time()

    # init data struct for svm
    #svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)
    
    # start training
    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # Iteration termination condition:
    #     Condition 1: reach max iteration
    #     Condition 2: no alpha changed after going through all samples,
    #                  in other words, all alpha (samples) fit KKT condition
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0

        # update alphas over all training examples
        if entireSet:
            for i in range(svm.data_num):
                alphaPairsChanged += updateParam(svm, i)
            print( '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1
        # update alphas over examples where alpha is not 0 & not C (not on boundary)
        else:
            nonBoundAlphasList = nonzero((svm.a > 0) * (svm.a < svm.c))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += updateParam(svm, i)
            print ('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1

        # alternate loop over all examples and non-boundary examples
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    #print ('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
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
    mySVM = SVM(data[0:80,0:2], data[0:80,2], 50, 1, 0.001, 'linear')
    #print('kmat:',mySVM.K)
    mySVM = smoTrain(mySVM, 50)
    #print(mySVM.a)
    w, b = calcWB(mySVM)
    for i in range(80):
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