#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import jieba
import re



with open('stop_word.txt') as f:
    stopwords = f.readlines()
    stop_list = []
    for i in stopwords:
        stop_list.append(i.replace("\n", ""))


"""
去除txt文本中的空格、数字、特定字母等
#去掉文本行里面的空格、换行\t、数字（其他有要去除的也可以放到' \t1234567890'里面）
"""
def remove_other(line):
    lines = filter(lambda ch: ch not in ' \t1234567890☝✌ ☺', line)
    return ''.join(lines)


"""
去掉标点符号(非文本部分)
#中文标点 ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.
#英文标点 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
"""
def remove_punctuation(line):
    try:
        line = re.sub(
            "[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+",
            "", line)
    except Exception as e:
        print(e)
    return line


def cleanData(sentence):
    _sentence = remove_other(sentence)
    _sentence = remove_punctuation(_sentence)
    setlast = jieba.lcut(_sentence, cut_all=False)
    seg_list = [i.lower() for i in setlast if i not in stop_list]
    return " ".join(seg_list)

def calculateSimilarity(sentence, doc_sentence_list):
    if doc_sentence_list == []:
        return 0
    vocab = {}
    for word in sentence.split():
        vocab[word] = 0  # 生成所在句子的单词字典，值为0

    docInOneSentence = ''
    for t in doc_sentence_list:
        docInOneSentence += (t + ' ')
        for word in t.split():
            vocab[word] = 1
    cv = CountVectorizer(vocabulary=vocab.keys())
    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]


def paper_process(texts):
    clean = []
    originalSentenceOf = {}
    if isinstance(texts,str):
        texts = [texts]
        print(texts)
    for line in texts:
        parts = line.split('。')  # 句子拆分
        for part in parts:
            part = re.sub(r'^\s+$', '', part)
            part = part.replace('\n','')
            if part == '':
                continue
            cl = cleanData(part)  # 句子切分以及去掉停止词
            clean.append(cl)  # 干净有重复的句子
            originalSentenceOf[cl] = part  # 字典格式
    setClean = set(clean)  # 干净无重复的句子

    # calculate Similarity score each sentence with whole documents
    scores = {}
    for data in clean:
        temp_doc = setClean - set([data])  # 在除了当前句子的剩余所有句子
        score = calculateSimilarity(data, list(temp_doc))  # 计算当前句子与剩余所有句子的相似度
        scores[data] = score  # 得到相似度的列表
    # print(scores)

    n = 1 * len(setClean) / 10  # 摘要的比例大小
    alpha = 0.9
    summarySet = []
    res_lst = []
    while n > 0:
        mmr = {}
        for sentence in scores.keys():
            if not sentence in summarySet:
                mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence,
                                                                                             summarySet)  # 公式
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        res_lst.append(originalSentenceOf[selected])
        n -= 1

    # print(res_lst)
    _str_res = '。'.join(res_lst) + '。' + '\n'
    return _str_res

if __name__ == '__main__':

    # with open('paper.txt') as f:
    #     current_labeled = f.readlines()
    # for i in current_labeled:
    #     if i != '\n':
    #         paper_process(i)

    _text = '安徽省五河县推出“医养结合+集中供养”，打造特困人员养护新模式。新华社记者 刘军喜摄 在刚刚结束的全国两会上，养老服务成为热词。《政府工作报告》把养老服务产业发展摆在重要位置，同时将其视为促进形成强大国内市场，持续释放内需潜力的重要抓手。当前，围绕我国2.5亿老年人吃、穿、住、用、医等方面的服务产业正在迅速兴起，多元化的养老服务体系也在逐渐形成，在相关服务业发展的环境下，老年人的晚年生活将更加安稳幸福—— 今年的《政府工作报告》提出，要大力发展养老特别是社区养老服务业，对在社区提供日间照料、康复护理、助餐助行等服务的机构给予税费减免、资金支持、水电气热价格优惠等扶持，新建居住区应配套建设社区养老服务设施。社区养老服务业应如何落实《政府工作报告》中的要求，养老服务业应当如何与时俱进?记者作了一番走访调查。 医养结合渐入佳境 在养老服务业发展过程中，医养结合正在扮演重要角色。 江苏省常州市是我国较早进入老龄化社会的城市之一。截至2017年年底，常州市60周岁及以上老年人口占总人口的比例就达到23.1%，老龄化、高龄化、空巢化、失能化趋势加快显现。 面对这么大规模的老年群体，如何提升养老服务水平?常州市的做法是在养老机构内设医疗机构、以医疗机构延伸养老服务，使养老与医护相互融合。 对广大失能半失能老人来说，专业照料和护理十分重要。在常州金东方护理院，记者注意到，护理院精心设计了单人间、双人间和三人间，以满足不同老年人的需要。每位入住金东方的老人都有一张卡，里面存储着老人的信息档案，包括健康情况、住院信息、饮食记录等，作为保证老年人健康安全的依据。同时，护理院还有医生、护士、康复师、营养师、社工、心理咨询师、护理员组成的多学科团队，全程负责老人的护理。 “医养结合不是一件容易的事情，它涉及社会保障制度、管理机制、服务体系等一系列问题。让老年人方便就医，是医养结合的关键。这不仅是空间布局的结合，还要在政策、机制、信息、文化等方面全面贯通。”金东方护理院董事长金建勇表示，就金东方而言，通过在社区里专门设立医院并由三甲医院来托管，提升了社区医疗保障水平，老人能够随时就近诊疗，享受便捷的医疗服务。 在位于常州市金坛区桑榆堂日间照料中心，记者同样感受到了“社区嵌入式”医养结合的魅力。这里除了为周边社区老年人提供日间照料、24小时居家照护、上门助浴外，还与江南医院下设的城区门诊部实现了联动，为老人们提供专项健康检查、康复训练和健康饮食服务。 “老人入住后，中心的专职医护人员会根据不同老人的具体情况，分别评估健康状况，最终制定出个性化的健康饮食计划，融入老人的护理计划中。”桑榆堂日间照料中心护理部主任告诉经济日报记者，专职营养师会定期根据老人们不同的身体健康状况制定饮食贴士并调整食谱。与此同时，医护人员会根据老人的身体状况定时定量分发药物，日间照料中心与门诊部密切互动，让老人拿药问诊更方便，省去了来回奔波的烦恼。 养老消费需求巨大 除了健康服务，撬动养老消费的另一把钥匙，便是老年日用品。在传统观念里，不少人会把老龄日用品等同于保健品以及衣服、鞋子等。事实上，随着老龄消费市场与时俱进，老年人对个性化、品质化、多元化的产品需求也越来越强烈。 可随意弯曲的汤勺、自带led灯的放大镜、可折叠防滑手杖、各类防褥疮垫、大型助浴设备、分指握力器……在常州国际健康养老产品展示中心，一系列更注重细节设计的小物件，解决了老年人生活中遇到的大问题。 “以养老用品作为切入点，能够推动整个养老全产业链的完善和发展。比如，养老教育培训、养老运营服务理念的引进，老年餐饮、适老化家庭改造，等等。”常州国际健康养老产品展示中心总经理林伟说。 随着越来越多的城市加入长期照护保险试点，除了以购买产品和服务为内容的健康养老消费，以长期护理保险为代表的金融类、保险类健康养老消费也在各地日渐兴起。从今年起，常州市联合太平洋、泰康等四家保险公司，发展“金融+健康养老”“保险+健康养老”等模式，试点推行长期护理保险制度，保费由政府补助、医保基金划转和个人缴费三部分组成，重点减轻重度失能人员基本生活照料的费用负担。 “健康养老服务不仅是事业，更是产业，要有意识、有规划、有举措地把健康养老服务业打造成新的经济引擎。”全国老龄办副主任吴玉韶表示。 智慧养老提升品质 尽管当前健康养老服务体系建设取得了一定成效，但城乡之间不够均衡，农村养老服务投入和建设滞后于城镇。与此同时，护理型养老机构一床难求，一些养老机构服务专业化水平较低的情况依然存在。 专家指出，随着社会发展和生活水平的提高，单纯的养老服务已经满足不了市场需求，老年人对医生的依赖和需求更高，养老需求呈现高、精、准发展态势。 在“互联网+”时代，大健康、大数据正推动互联网与健康养老产业加速融合发展。多地运用互联网、物联网和大数据等信息技术，推动健康养老向信息化、智能化方向发展。吴玉韶认为，医养结合的“医”，更多的是指健康管理，健康管理是大数据、动态的，需要新技术加以承载。 在常州市天宁区智慧养老服务指挥平台大厅，接线员正在紧张有序地接听有各种服务需求的电话。通过建立“老人—子女—平台”三方联动制度，用户可以通过app、电话、短信及上门服务等形式，自主选择服务。 “该平台不仅可以实现下单服务，还可以监督反馈，保证服务质量。”常州市卫健委宣传处陆人杰透露，通过平台，子女可以更便捷地关注父母日常生活情况，并对相关老人服务提出改进意见。 目前，智能健康设备也开始进入寻常百姓家，并在养老领域中广泛应用。比如，常州市富强新村社区的改造项目，健康智慧养老服务体系已经融入居民日常生活中，全自动体检设备、智能养老机器人等适老用品一应俱全。这些智能设备可以实现建档老人一对一的健康风险评估，并制定促进老年人诊疗的个性化方案。 常州市副市长陈正春认为，健康养老产业的兴起发展，政府在其中要起到扶持、指导、催化作用。“我们将充分运用云计算、大数据、物联网、移动互联网、人工智能等新技术，积极探索发展‘互联网+健康养老’等新兴业态，拓展健康养老产业的发展空间。”陈正春说。'

    res = paper_process(_text)
    print(res)