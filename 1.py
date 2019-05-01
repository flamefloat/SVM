import math
import numpy as np 
a=math.exp(0)
b=np.array([1,2,3,3])
c=np.array([9,1,1,7])

d=b-c
e=np.linalg.norm(d,ord=2)
pos = np.argmax(c)
print(pos)
print(2>=1 and 2<=3)