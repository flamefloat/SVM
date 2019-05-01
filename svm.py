import math
import numpy as np 

class SVM():
    """
    data:训练数据集，NxM矩阵，实列数：N,特征向量维度：M
    label:标签，1xN的矩阵，取值为-1，1
    c:惩罚系数
    sigma:RBF核函数参数
    """
    def __init__(self, data, label, c, sigma):
        self.data = data
        self.label = label
        self.c = c
        self.sigma = sigma
        self.data_num = data.shape[0] # 训练集长度
        self.data_width = data.shape[1] #实例维度
        self.a = np.zeros(data.shape[0])
        self.b = 0

    def k(self, x1, x2): # RBF核函数
        return math.exp(-sum((x1-x2)**2)/(2*self.sigma**2))

    
    def g(self,x): # 函数g(x)
        temp = 0
        for i in range(self.data_num):
            temp += self.a[i]*self.label[i]*self.k(self.data[i,:],x)
        temp += self.b
        return temp

    def finishTrain(self, a, b): # 判断是否满足结束训练的条件

        return done



    def smoTrain(self):#采用SMO算法
        E = np.zeros(self.data_num)
        for i in range(self.data_num): # 初始化E值
            E[i] = self.g(self.data[i,:]) - self.label[i]
        error = np.zeros(self.data_num)
        while True:
            for j in range(self.data_num): # 外层循环，确定a1
                if self.a[j] > 0 and self.a[j] < self.c:
                    flag = self.label[j]*self.g(self.data[j,:])
                    if flag == 1:
                        continue
                    else:
                        error[j] = abs(1-flag)
            if max(error) == 0: # 所有0<a<c的样本点均满足KKT条件,遍历整个训练集
                for j in range(self.data_num): 
                    if self.a[j] == 0:
                        flag = self.label[j]*self.g(self.data[j,:])
                        if flag >=1:
                            continue
                        else:
                            error[j] = abs(1-flag)
                    elif self.a[j] == self.c:
                        flag = self.label[j]*self.g(self.data[j,:])
                        if flag <=1:
                            continue
                        else:
                            error[j] = abs(1-flag)
            pos_j = np.argmax(error) # pos_i:a2, pos_j: a1
            if E[pos_j] >=0:
                pos_i = np.argmin(E) #TODO a1=a2?
            else:
                pos_i = np.argmax(E)
            #更新 a1, a2, b,E
            old_a1 = self.a[pos_j]
            old_a2 = self.a[pos_i]
            eta = self.k(self.data[pos_j], self.data[pos_j]) + self.k(self.data[pos_i], self.data[pos_i])\
                    -2*self.k(self.data[pos_i], self.data[pos_j])
            temp_a2 = old_a2 + self.label[pos_i] * (E[pos_j] - E[pos_i]) / eta
            if self.label[pos_i] == self.label[pos_j]:
                L = max(0, old_a2 + old_a1 - self.c)
                H = min(self.c, old_a2 + old_a1)
            else:
                L = max(0, old_a2 - old_a1)
                H = min(self.c, self.c + old_a2 - old_a1)
            if temp_a2 > H:
                self.a[pos_i] = H
            elif temp_a2 < L:
                self.a[pos_i] = L
            else:
                self.a[pos_i] = temp_a2
            self.a[pos_j] = old_a1 + self.label[pos_j] * self.label[pos_i] * (old_a2 - self.a[pos_i])
            new_b1 = -E[pos_j]\
                        - self.label[pos_j]*self.k(self.data[pos_j], self.data[pos_j])*(self.a[pos_j]-old_a1)\
                        - self.label[pos_i]*self.k(self.data[pos_i], self.data[pos_j])*(self.a[pos_i]-old_a2)\
                        + self.b
            new_b2 = -E[pos_i]\
                        - self.label[pos_j]*self.k(self.data[pos_j], self.data[pos_i])*(self.a[pos_j]-old_a1)\
                        - self.label[pos_i]*self.k(self.data[pos_i], self.data[pos_i])*(self.a[pos_i]-old_a2)\
                        + self.b
            self.b = (new_b1 + new_b2) / 2
            E[pos_j] = self.g(self.data[pos_j,:]) - self.label[pos_j]
            E[pos_i] = self.g(self.data[pos_i,:]) - self.label[pos_i]
            done = self.finishTrain(self.a, self.b)
            if done:
                break
        return self.a, self.b
            



            






