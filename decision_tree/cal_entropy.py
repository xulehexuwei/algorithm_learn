#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''                                                          
Copyright (C)2018 SenseDeal AI, Inc. All Rights Reserved                                                      
Author: xuwei                                        
Email: weix@sensedeal.ai                                 
Description:                                    
'''

from math import log


def calEntropy(dataSet):
    """
    :param dataSet: [[*value, label], [1, 'x']]
    :return:
    """
    numEntries = len(dataSet)  # 样本数
    labelCounts = {}  # 该数据集每个类别的频数
    for featVec in dataSet:  # 对每一行样本
        currentLabel = featVec[-1]  # 该样本的标签
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDatSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDatSet.append(reducedFeatVec)
    return retDatSet


def getBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for axis in range(numFeatures):
        value_set = set([featVec[axis] for featVec in dataSet])
        newEntropy = 0.0
        for value in value_set:
            subDatSet = splitDataSet(dataSet, axis, value)
            prob = len(subDatSet) / float(len(dataSet))
            newEntropy += prob * calEntropy(subDatSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = axis
    return bestFeature


if __name__ == '__main__':
    dataSet = [[1, 'x'], [2, 'w'], [3, 'x'], [4, 'x']]
    r = calEntropy(dataSet)
    print(r)

    dataSet = [[1, 2, 'x'], [3, 2, 'w'], [4, 2, 'x']]
    r = calEntropy(dataSet)
    print(r)

    r = splitDataSet(dataSet, 0, 1)
    print(r)

    r = getBestFeatureToSplit(dataSet)
    print(r)