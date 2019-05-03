import scipy.io as scio
import matplotlib.pyplot as plt
D1 = scio.loadmat('C:\\Users\\MH\\Desktop\\MyCode\\MLalgorithm\\SVM\\Data2.mat')
data = D1['data']
for i in range(data.shape[0]):
    if data[i,2] == 1:
        plt.plot(data[i, 0], data[i, 1], 'or')
    if data[i,2] == -1:
        plt.plot(data[i, 0], data[i, 1], 'ob')
plt.show()