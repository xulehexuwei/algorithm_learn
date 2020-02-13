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
    :param dataSet: [(value, label), (1, 'x')]
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


if __name__ == '__main__':
    dataSet = [(1, 'x'), (2, 'w'), (3, 'x'), (4, 'x')]
    r = calEntropy(dataSet)
    print(r)

    dataSet = [(1, 'x'), (3, 'x'), (4, 'x')]
    r = calEntropy(dataSet)
    print(r)
