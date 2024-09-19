import numpy as np


s = np.array([[1,2],[3,4],[5,6]])
s1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
Y = np.array([1,-1,2,-2])
y = np.array([1,1,2,-2,-1])
X = np.array([[1,'S'] , [2,'S'] , [3,'M'] , [2,'L'] , [1,'L']])
ch1 = np.array([1,2,3])
ch2 = np.array(['S','M','L'])

print(s.shape)
print(s1.shape)
print(Y.shape)
print(y.shape)

s1[1] = s1[1]/2
print(s1)

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
print('yfrequency: ',yfrequency)
print('yprior: ',yprior)

N = X.shape[0]
K = yfrequency.shape[0]
C = ch1.shape[0]  # C为特征chara所有可能取值的个数
i = 1
cmid = np.zeros((K, C),dtype=float)
print(N,K,C)
# print(ch1[1])
# print(X[1][i-1])

for j in range(0,C):
    chj = ch1[j]
    for k in range(0,K):
        for n in range(0,N):
            if chj == eval(X[n][i-1]):
                if Y[k] == y[n]:
                    cmid[k][j] += 1
print(cmid)
cfre = np.sum(cmid,axis=1)
for k in range(0, K):
    kfre = cfre[k]
    cmid[k] = cmid[k]/kfre





