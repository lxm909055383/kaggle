#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


#读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  #六个分类

#数据预处理
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
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
        # 将非字母正则替换为空格
        text = re.sub(r"[^a-z]", " ", text)
        #添加到评论列表
        comment_list.append(text)
    return comment_list

train["clean_comment_text"] = clean_text(train['comment_text'])
test["clean_comment_text"] = clean_text(test['comment_text'])

all_comment_list = list(train['clean_comment_text']) + list(test['clean_comment_text'])
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', token_pattern=r'\w{1,}',
                         max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_vec = text_vector.transform(train['clean_comment_text'])
test_vec = text_vector.transform(test['clean_comment_text'])
# print(train_vec)

x_train, x_valid, y_train, y_valid = train_test_split(train_vec, train[labels], test_size=0.1, random_state=2018)
x_test = test_vec