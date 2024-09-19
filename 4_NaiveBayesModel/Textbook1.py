import numpy as np


# 定义指示函数
def Indc(x,y):
    if x == y:
        return 1
    else:
        return 0


# 定义先验概率，y是一个narray数组，Y是类标记的集合，lambda大于等于0
# yprior中的第k个元素代表类标记Y = ck的先验概率
def PriorPro(y , Y ):
    N = y.shape[0]
    K = Y.shape[0]
    ymid = np.zeros((K, N))
    for k in range(0, K):
        ck = Y[k]
        for n in range(0, N):
            if ck == y[n]:
                ymid[k, n] = 1
    yfrequency = np.sum(ymid,axis=1)
    yprior = np.average(ymid,axis=1)
    return yfrequency,yprior


# 求X关于第i个特征的条件概率
def CharaProSim(X,chara,y,Y,i):
    N = X.shape[0]
    K = Y.shape[0]
    CVN = chara.shape[0]  # C为特征chara所有可能取值的个数
    cmid = np.zeros((K, CVN),dtype=float)
    for j in range(0, CVN):
        chj = chara[j]
        for k in range(0, K):
            for n in range(0, N):
                if type(chj) == np.int32:
                    if chj == eval(X[n][i - 1]):
                        if Y[k] == y[n]:
                            cmid[k][j] += 1
                else:
                    if chj == X[n][i - 1]:
                        if Y[k] == y[n]:
                            cmid[k][j] += 1
    cfre = np.sum(cmid,axis=1)
    for k in range(0, K):
        kfre = cfre[k]
        cmid[k] = cmid[k]/kfre
    return cmid


# 定义条件概率，x是实例，X是输入，chara是特征取值的集合
def CondPro(x,X,chara,Y,y):
    N,CN = X.shape   # N是样本的个数，CN是特征的个数
    K = Y.shape[0]
    ConP = np.zeros((K,CN),dtype=float)
    for n in range(0,CN):
        chn = chara[n+1]  # 取第n个特征的所有可能取值
        xch = x[n]  # 实例x的第n个特征的取值
        i = 0
        while chn[i] != xch:
            i += 1
        xn = CharaProSim(X, chn, y, Y, n+1)[:,i]
        ConP[:,n] = xn
    return ConP


# 朴素贝叶斯分类器
def NBclassifier(Y,yprior,ConP):
    ConP = np.cumprod(ConP,axis=1)[:,-1].T
    mid = list(ConP*yprior)
    ind = mid.index(max(mid))
    return Y[ind]


X = [[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],[2,'M'],[2,'M'],[2,'L'],[2,'L'],[3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']]
X = np.array(X)
Y = np.array([1,-1])
y = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
y = np.array(y)
x = [2,'S']
chara = {1:np.array([1,2,3]) , 2:np.array(['S','M','L'])}
yfrequency,yprior = PriorPro(y , Y )
ConP = CondPro(x,X,chara,Y,y)
fx = NBclassifier(Y,yprior,ConP)
print(f'实例x的分类是{fx}')






