#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re

#读取数据
trainContent = pd.read_csv('train.csv')
testContent = pd.read_csv('test.csv')
sampleContent = pd.read_csv('sample_submission.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  #六个分类

#数据预处理
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text)

        # 恢复常见的简写
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)

        # 恢复特殊符号的英文单词
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')

        comment_list.append(text)
    return comment_list