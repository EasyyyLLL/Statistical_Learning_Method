from typing import ItemsView
import numpy as np

obj1 = np.array([1,2,3])
obj2 = np.array([4,5,6])
diff = obj1 - obj2
print(diff)
print(np.dot(diff,diff))

x = np.array([[1, 0], [2, 0], [3, 1]], np.int32)
print(x)


adict = {'a':2,'b':3,'c':1,'d':4}
blist = [('a',2) , ('b',3) , ('c',1) , ('d',4)]
adlist = list(adict.items())
print(adlist)
adlist.sort(key = lambda x : x[1])
print(adlist)

