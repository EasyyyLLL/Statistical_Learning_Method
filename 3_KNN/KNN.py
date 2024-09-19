import numpy as np
import math
import operator


# 计算点与点之间的距离
def distance(object1 , object2):
    obj1 = np.array(object1)
    obj2 = np.array(object2)
    diff = obj1 - obj2
    L = math.sqrt(np.dot(diff,diff))
    return L


# 获得目标点的k个邻居
def getneigh(traindata , target , k):
    traindatanp = np.array(traindata)
    tarnp = np.array(target)
    distanceall = []
    for x in traindatanp:
        Lx = distance(x,tarnp)
        distanceall.append((x,Lx))
    distanceall.sort(key = lambda x : x[1] , reverse=False)
    neighbors = []
    for i in range(k):
        neighbors.append(distanceall[i][0])
    return neighbors


# 获取k个邻居的类别并投票决定目标点的类别
def getclass(neighborslabel):
    classall = {}
    for x in neighborslabel:
        label = x[-1]
        if label in classall:
            classall[label] += 1
        else:
            classall[label] = 1
    classallitem = list(classall.items())
    classallitem.sort(key = lambda x : x[1] , reverse=False)
    return classallitem[0][0]


# 计算准确率
def getAccuracy(testset, pridiction):
    correct = 0
    for k in range(len(testset)):
        testset[k] = pridiction[k]
        correct += 1
    return correct/len(testset)*100.0



xdata = [[3, 104] , [2, 100] , [1, 81] , [101, 10] , [99, 5] , [98, 2]]
ylabel = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
target = [18,90]
k = 3
neighbors = getneigh(xdata,target,k)
print(neighbors)
neighborslabel = []
for x in neighbors:
    xlist = list(x)
    xindex = xdata.index(xlist)
    xlabel = ylabel[xindex]
    xlist.append(xlabel)
    neighborslabel.append(xlist)
print(neighborslabel)
classtarget = getclass(neighborslabel)
print(classtarget)


