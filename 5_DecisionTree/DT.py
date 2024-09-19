# 导入模块
# 用pandas模块的read_csv()函数读取数据文本；用numpy模块将dataframe转换为list（列表）；用Counter来完成计数；用math模块的log2函数计算对数。
import numpy as np
import pandas as pd
import math
import collections


# 导入数据
def import_data():
    data = pd.read_csv('D://PythonFiles//StatisticalLearningMethod//5_DecisionTree//watermelon.txt')
    data.head(10)
    data=np.array(data).tolist()
    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']

    # 特征对应的所有可能的情况
    labels_full = {}

    for i in range(len(labels)):
        labelList = [example[i] for example in data]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel
    return data,labels,labels_full


# 调用函数获取数据
data,labels,labels_full=import_data()


# 计算初始的信息熵，就是不分类之前的信息熵值
def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵(香农熵)
    :param dataSet:
    :return:
    """
    # 计算出数据集的总数
    numEntries = len(dataSet)

    # 用来统计标签
    labelCounts = collections.defaultdict(int)

    # 循环整个数据集，得到数据的分类标签
    for featVec in dataSet:
        # 得到当前的标签
        currentLabel = featVec[-1]

        # # 如果当前的标签不再标签集中，就添加进去（书中的写法）
        # if currentLabel not in labelCounts.keys():
        #     labelCounts[currentLabel] = 0
        #
        # # 标签集中的对应标签数目加一
        # labelCounts[currentLabel] += 1

        # 也可以写成如下
        labelCounts[currentLabel] += 1

    # 默认的信息熵
    shannonEnt = 0.0

    for key in labelCounts:
        # 计算出当前分类标签占总标签的比例数
        prob = float(labelCounts[key]) / numEntries

        # 以2为底求对数
        shannonEnt -= prob * math.log2(prob)

    return shannonEnt


# 查看初始信息熵
print('初始信息熵')
print(calcShannonEnt(data)) # 输出为：0.9975025463691153
print('*'*50)

# 获取每个特征值的数量，这是为后面计算信息增益做准备
def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征值，将数据集划分
    :param dataSet: 数据集
    :param axis: 给定特征值的坐标
    :param value: 给定特征值满足的条件，只有给定特征值等于这个value的时候才会返回
    :return:
    """
    # 创建一个新的列表，防止对原来的列表进行修改
    retDataSet = []

    # 遍历整个数据集
    for featVec in dataSet:
        # 如果给定特征值等于想要的特征值
        if featVec[axis] == value:
            # 将该特征值前面的内容保存起来
            reducedFeatVec = featVec[:axis]
            # 将该特征值后面的内容保存起来，所以将给定特征值给去掉了
            reducedFeatVec.extend(featVec[axis + 1:])
            # 添加到返回列表中
            retDataSet.append(reducedFeatVec)

    return retDataSet


# 计算信息增益来确定最好的数据集划分
def chooseBestFeatureToSplit(dataSet, labels):
    """
    选择最好的数据集划分特征，根据信息增益值来计算
    :param dataSet:
    :return:
    """
    # 得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1

    # 计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 基础信息增益为0.0
    bestInfoGain = 0.0

    # 最好的特征值
    bestFeature = -1

    # 对每个特征值进行求信息熵
    for i in range(numFeatures):
        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]

        # 将当前特征唯一化，也就是说当前特征值中共有多少种
        uniqueVals = set(featList)

        # 新的熵，代表当前特征值的熵
        newEntropy = 0.0

        # 遍历现在有的特征的可能性
        for value in uniqueVals:
            # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
            subDataSet = splitDataSet(dataSet=dataSet, axis=i, value=value)

            # 计算出权重
            prob = len(subDataSet) / float(len(dataSet))

            # 计算出当前特征值的熵
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 计算出“信息增益”
        infoGain = baseEntropy - newEntropy

        #print('当前特征值为：' + labels[i] + '，对应的信息增益值为：' + str(infoGain)+"i等于"+str(i))

        #如果当前的信息增益比原来的大
        if infoGain > bestInfoGain:
            # 最好的信息增益
            bestInfoGain = infoGain
            # 新的最好的用来划分的特征值
            bestFeature = i

    #print('信息增益最大的特征为：' + labels[bestFeature])
    return bestFeature


# 判断各个样本集的各个属性是否一致
def judgeEqualLabels(dataSet):
    """
    判断数据集的各个属性集是否完全一致
    :param dataSet:
    :return:
    """
    # 计算出样本集中共有多少个属性，最后一个为类别
    feature_leng = len(dataSet[0]) - 1

    # 计算出共有多少个数据
    data_leng = len(dataSet)

    # 标记每个属性中第一个属性值是什么
    first_feature = ''

    # 各个属性集是否完全一致
    is_equal = True

    # 遍历全部属性
    for i in range(feature_leng):
        # 得到第一个样本的第i个属性
        first_feature = dataSet[0][i]

        # 与样本集中所有的数据进行对比，看看在该属性上是否都一致
        for _ in range(1, data_leng):
            # 如果发现不相等的，则直接返回False
            if first_feature != dataSet[_][i]:
                return False

    return is_equal


# 绘制决策树（字典）
def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 特征标签
    :return:
    """
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签，或者所有的样本在全部属性上都一致
    if len(dataSet[0]) == 1 or judgeEqualLabels(dataSet):
        # 返回剩下标签中出现次数较多的那个
        return majorityCnt(classList)

    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet=dataSet, labels=labels)
    print(bestFeat)
    # 得到最好特征的名称
    bestFeatLabel = labels[bestFeat]
    print(bestFeatLabel)
    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    myTree = {bestFeatLabel: {}}

    # 将本次划分的特征值从列表中删除掉
    del(labels[bestFeat])

    # 得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]

    # 唯一化，去掉重复的特征值
    uniqueVals = set(featValues)

    # 遍历所有的特征值
    for value in uniqueVals:
        # 得到剩下的特征标签
        subLabels = labels[:]
        subTree = createTree(splitDataSet(dataSet=dataSet, axis=bestFeat, value=value), subLabels)
        # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
        myTree[bestFeatLabel][value] = subTree
    return myTree


# 调用函数并打印，就可以看到一个字典类型的树了
mytree=createTree(data,labels)
print(mytree)





