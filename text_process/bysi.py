from numpy import *

def loadDataSet():
    '''
    postingList: 进行词条切分后的文档集合
    classVec:类别标签
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # postingList = [['个股方面', '股价大涨24%', '联合健康股价大涨近4%', '后者将斥资128亿美元现金收购前者'],
    #                ['资金涌港', '追捧H股', '港股通净买入额', '创历史新高'],
    #                ['一汽集团收购后', '一汽夏利', '并没能够', '获得集团更多的资源共享'],
    #                ['旗下合资公司中', '股权结构复杂', '，收购操作', '难度较大'],
    #                ['从成交金额来看', '港股通买入成交42.03亿元', '卖出成交13.90亿元', '总成交55.93亿元，也为历史上最高'],
    #                ['不过期间高盛“套现”了', '截至2014年12月31日', '其对宋河的持股', '下降至17.72%']]
    # classVec = [0, 0, 1, 1, 0, 1]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])#使用set创建不重复词表库
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建两个集合的并集
    print(list(vocabSet))
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)#创建一个所包含元素都为0的向量
    #遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
'''
我们将每个词的出现与否作为一个特征，这可以被描述为词集模型(set-of-words news_model)。
如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息,
这种方法被称为词袋模型(bag-of-words news_model)。
在词袋中，每个单词可以出现多次，而在词集中，每个词只能出现一次。
为适应词袋模型，需要对函数setOfWords2Vec稍加修改，修改后的函数称为bagOfWords2VecMN
'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数(此处仅处理两类分类问题)
    trainMatrix:文档矩阵
    trainCategory:每篇文档类别标签
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Num = ones(numWords); p1Num = ones(numWords)#
    p0Denom = 2.0; p1Denom = 2.0 #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num/p1Denom)#change to log()
    p0Vect = log(p0Num/p0Denom)#change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    分类函数
    vec2Classify:要分类的向量
    p0Vec, p1Vec, pClass1:分别对应trainNB0计算得到的3个概率
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #训练模型，注意此处使用array
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


testingNB()