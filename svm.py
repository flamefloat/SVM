import math
import numpy as np 
import scipy.io as scio
import matplotlib.pyplot as plt

class SVM():
    """
    data:训练数据集，NxM矩阵，实列数：N,特征向量维度：M
    label:标签，1xN的矩阵，取值为-1，1
    c:惩罚系数
    sigma:RBF核函数参数
    """
    def __init__(self, data, label, c, sigma, maxIter, kernel):
        self.data = data
        self.label = label
        self.c = c
        self.sigma = sigma
        self.maxIter = maxIter
        self.kernel = kernel
        self.data_num = data.shape[0] # 训练集长度
        self.data_width = data.shape[1] #实例维度
        #self.a = np.zeros(data.shape[0])
        #self.b = 0
        #self.E = np.zeros(self.data_num)

    def k(self, x1, x2): # 默认RBF核函数
        if self.kernel == 'linear':
            return sum(x1*x2)
        else:
            return math.exp(-sum((x1-x2)**2)/(2*self.sigma**2))

    
    def g(self, x, a, b): # 函数g(x)
        temp = 0
        for i in range(self.data_num):
            temp += a[i]*self.label[i]*self.k(self.data[i,:],x)
        temp += b
        return temp

    def updateParam(self, a, b, E, pos_j, pos_i):
        #更新 a1, a2, b,E
        old_a1 = a[pos_j]
        old_a2 = a[pos_i]
        eta = self.k(self.data[pos_j], self.data[pos_j])\
             +self.k(self.data[pos_i], self.data[pos_i])\
             -2*self.k(self.data[pos_i], self.data[pos_j])
        temp_a2 = old_a2 + self.label[pos_i] * (E[pos_j] - E[pos_i]) / eta
        if self.label[pos_i] == self.label[pos_j]:
            L = max(0, old_a2 + old_a1 - self.c)
            H = min(self.c, old_a2 + old_a1)
        else:
            L = max(0, old_a2 - old_a1)
            H = min(self.c, self.c + old_a2 - old_a1)
        if temp_a2 > H:
            a[pos_i] = H
        elif temp_a2 < L:
            a[pos_i] = L
        else:
            a[pos_i] = temp_a2
        a[pos_j] = old_a1 + self.label[pos_j] * self.label[pos_i] * (old_a2 - a[pos_i])
        new_b1 = -E[pos_j]\
                 - self.label[pos_j]*self.k(self.data[pos_j], self.data[pos_j])*(a[pos_j]-old_a1)\
                 - self.label[pos_i]*self.k(self.data[pos_i], self.data[pos_j])*(a[pos_i]-old_a2)\
                 + b
        new_b2 = -E[pos_i]\
                 - self.label[pos_j]*self.k(self.data[pos_j], self.data[pos_i])*(a[pos_j]-old_a1)\
                 - self.label[pos_i]*self.k(self.data[pos_i], self.data[pos_i])*(a[pos_i]-old_a2)\
                 + b
        b = (new_b1 + new_b2) / 2
        E[pos_j] = self.g(self.data[pos_j,:], a, b) - self.label[pos_j]
        E[pos_i] = self.g(self.data[pos_i,:], a, b) - self.label[pos_i]
        return a, b, E

    def find(self, a, b, E, pos_j, error, last_loss):
        if E[pos_j] >=0:
            pos_i = np.argmin(E) #TODO 
        else:
            pos_i = np.argmax(E)
        temp_a, temp_b, temp_E = self.updateParam(a, b, E, pos_j, pos_i)
        next_loss = self.lossCount(temp_a)
        Temp_a = []
        Temp_b = []
        Temp_E = []
        print('lastloss:',last_loss,'nextloss:',next_loss)

        if last_loss > next_loss:
            return pos_j, pos_i
        else :
            tempLoss = []
            for i in range(self.data_num):#遍历整个训练集寻找a2
                if pos_j != i:
                    pos_i = i
                else:
                    tempLoss.append(10000)
                    Temp_a.append([0])
                    Temp_b.append([0])
                    Temp_E.append([0])
                    continue
                temp_a, temp_b, temp_E = self.updateParam(a, b, E, pos_j, pos_i)
                next_loss = self.lossCount(temp_a)
                tempLoss.append(next_loss)
                Temp_a.append(temp_a)
                Temp_b.append(temp_b)
                Temp_E.append(temp_E)
            pos_i = np.argmin(tempLoss)
            print('tempLoss[pos_i]',tempLoss[pos_i])
            if tempLoss[pos_i] < last_loss:
                return pos_j, pos_i
            else : # 寻找新的a1
               
                temp_error = error
                temp_error[pos_j] = 0
                pos_j = np.argmax(temp_error)
                print('**********','pos_j:',pos_j,'temp_error[pos_j]:',temp_error[pos_j])
                #print('lastloss:',last_loss)
                self.find(a, b, E, pos_j, temp_error, last_loss)
                 
            



    def finishTrain(self, a, b): # 判断是否满足结束训练的条件
        temp = 0
        for i in range(self.data_num):
            temp += a[i]*self.label[i]
        if temp != 0:
            return False
        else:
            for i in range(self.data_num):
                if self.label[i]*self.g(self.data[i,:], a, b) == 1:
                    if a[i] >= self.c or a[i] <= 0:
                        return False
                elif self.label[i]*self.g(self.data[i,:], a, b) < 1:
                    if a[i] != self.c:
                        return False
                else:
                    if a[i] != 0:
                        return False
        return True

    def lossCount(self, a):
        loss = 0
        for i in range(self.data_num):
            for j in range(self.data_num):
                loss += a[i]*a[j]*self.label[i]*self.label[j]\
                       *self.k(self.data[i,:],self.data[j,:])
        loss = 0.5*loss-sum(a)
        return loss


    def smoTrain(self):#采用SMO算法
        a = np.zeros(data.shape[0])
        b = 0
        iterCount = 0 # 迭代次数
        E = np.zeros(self.data_num)
        last_loss = self.lossCount(a) # 目标函数值
        for i in range(self.data_num): # 初始化E值
            E[i] = self.g(self.data[i,:], a, b) - self.label[i]
        #error = np.zeros(self.data_num)
        while True:
            iterCount += 1
            error = np.zeros(self.data_num)
            for j in range(self.data_num): # 外层循环，确定a1
                if a[j] > 0 and a[j] < self.c:
                    flag = self.label[j]*self.g(self.data[j,:], a, b)
                    if flag == 1:
                        continue
                    else:
                        error[j] = abs(1-flag)
            if max(error) == 0: # 所有0<a<c的样本点均满足KKT条件,遍历整个训练集
                for j in range(self.data_num): 
                    if a[j] == 0:
                        flag = self.label[j]*self.g(self.data[j,:], a, b)
                        if flag >1:
                            continue
                        else:
                            error[j] = abs(1-flag)
                    elif a[j] == self.c:
                        flag = self.label[j]*self.g(self.data[j,:], a, b)
                        if flag <1:
                            continue
                        else:
                            error[j] = abs(1-flag)
            pos_j = np.argmax(error) # 违反KKT条件最严重的为a1; pos_i:a2, pos_j: a1
            pos_j, pos_i = self.find(a, b, E, pos_j, error, last_loss)
            a, b, E = self.updateParam(a, b, E, pos_j, pos_i)
            next_loss = self.lossCount(a)

            #temp_E = E
            """
            if E[pos_j] >=0:
                pos_i = np.argmin(temp_E) #TODO 
            else:
                pos_i = np.argmax(temp_E)
            temp_a, temp_b, temp_E = self.updateParam(a, b, E, pos_j, pos_i)
            next_loss = self.lossCount(temp_a)
            Temp_a = []
            Temp_b = []
            Temp_E = []
            if last_loss <= next_loss:
                tempLoss = []
                for i in range(self.data_num):#遍历整个训练集寻找a2
                    if pos_j != i:
                        pos_i = i
                    else:
                        tempLoss.append(10000)
                        Temp_a.append([0])
                        Temp_b.append([0])
                        Temp_E.append([0])
                        continue
                    temp_a, temp_b, temp_E = self.updateParam(a, b, E, pos_j, pos_i)
                    next_loss = self.lossCount(temp_a)
                    tempLoss.append(next_loss)
                    Temp_a.append(temp_a)
                    Temp_b.append(temp_b)
                    Temp_E.append(temp_E)
                pos_i = np.argmin(tempLoss)
                if tempLoss[pos_i] < last_loss:
                    continue
                else : # 寻找新的a1
                    temp_error = error
                    temp_error[pos_j] = 0
                    pos_j = np.argmax[temp_error]
            """

                    


            """
            while True :
                if last_loss > next_loss:
                    break
                for i in range(self.data_num):#遍历边界上的支持向量点作为a2
                    if self.a[i] > 0 and self.a[i] < self.c and pos_j != i:
                        pos_i = i
                        self.updateParam(pos_j, pos_i)
                        next_loss = self.lossCount()
                        if last_loss > next_loss:
                            break 
                if last_loss <= next_loss:#边界上无合适的a2，遍历整个训练集作a2
                    TempCount = 0
                    for i in range(self.data_num):
                        if pos_j != i:
                            pos_i = i
                        else:
                            continue
                        self.updateParam(pos_j, pos_i)
                        next_loss = self.lossCount()
                        TempCount += 1
                        if last_loss > next_loss:
                            break
                        if TempCount == self.data_num - 1:#重新选择a1
                            temp_error[pos_j] = 0
                            pos_j = np.argmax(temp_error)
                if sum(temp_error) == 0:
                    break
            """
            last_loss = next_loss
            print('iter:',iterCount,'a1:',pos_j,'a2:',pos_i)

            #判断是否结束训练
            #done = self.finishTrain(self.a, self.b)
            if (iterCount > self.maxIter):
                done = self.finishTrain(a, b)
                break
        #return self.a, self.b
        w = np.zeros(self.data_width)
        temp_B = 0
        pos = 0
        for i in range(self.data_num):
            w += a[i]*self.label[i]*self.data[i,:]
            if a[i] > 0 and a[i] < self.c:
                pos = i
        print('w：',w)
        print('pos:',pos)
        for i in range(self.data_num):
            temp_B += a[i]*self.label[i]*self.k(self.data[i,:], self.data[pos,:])
        B = self.label[pos] - temp_B
        return w, B, a, done


if __name__ == '__main__':
    D1 = scio.loadmat('C:\\Users\\MH\\Desktop\\MyCode\\MLalgorithm\\SVM\\Data1.mat')
    data = D1['data']
    #print(data[1:5,:])
    mySVM = SVM(data[0:70,0:2], data[0:70,2], 10, 1, 50, 'linear')
    w, b, a, done = mySVM.smoTrain()
    print(done)
    print(a)
    #print(w)
    for i in range(70):
        if a[i] > 0:
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



            






