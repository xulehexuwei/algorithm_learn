#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''                                                          
Copyright (C)2018 SenseDeal AI, Inc. All Rights Reserved                                                      
Author: xuwei                                        
Email: weix@sensedeal.ai                                 
Description:                                    
'''


import fastText

# #训练模型
# classifier = fastText.train_supervised("./data/fasttext_data.txt",
#                                   label="__label__",
#                                   lr=0.05,
#                                   epoch=10,
#                                   dim=50,
#                                   loss='ns',
#                                   ws=5)
#
#
# classifier.save_model('./news_model/fasttext.news_model')

model = fastText.load_model('./news_model/fasttext.news_model')

# test = news_model.test("./data/fasttext_data.txt")

# test = news_model.test_label("./data/fasttext_data.txt")

test = model.predict('中国巨石股票上市20周年庆典暨智能制造基地项目投产仪式在浙江桐乡举行',k=3)

print(test)
