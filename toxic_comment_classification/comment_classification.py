#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#读取数据
train = pd.read_csv('train.csv',nrows = 1000)
test = pd.read_csv('test.csv',nrows = 1000)
sample_submission = pd.read_csv('sample_submission.csv',nrows = 1000)
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  #六个分类

#数据预处理
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
        #将缩写词去掉
        text = re.sub(r"[a-z]*'[a-z]*", "", text)
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

#将训练集分为训练数据和验证数据(x_train训练集数据，x_valid验证集数据，y_train训练集类别，y_valid验证集类别，)
x_train, x_valid, y_train, y_valid = train_test_split(train_vec, train[labels], test_size=0.1, random_state=2018)
x_test = test_vec

accuracy = []
for label in labels:
    clf = LogisticRegression(C=6)
    clf.fit(x_train, y_train[label])
    y_pre = clf.predict(x_valid)
    train_scores = clf.score(x_train, y_train[label])
    valid_scores = accuracy_score(y_pre, y_valid[label])
    print("{} train score is {}, valid score is {}".format(label, train_scores, valid_scores))
    accuracy.append(valid_scores)
    pred_proba = clf.predict_proba(x_test)[:, 1]
    sample_submission[label] = pred_proba
print("Total cv accuracy is {}".format(np.mean(accuracy)))