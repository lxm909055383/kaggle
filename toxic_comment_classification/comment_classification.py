#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


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
        #将缩写词去掉
        text = re.sub(r"[a-z]*'[a-z]*", "", text)
        # 将非字母正则替换为空格
        text = re.sub(r"[^a-z]", " ", text)
        #添加到评论列表
        comment_list.append(text)
    return comment_list

train["clean_comment_text"] = clean_text(train['comment_text'])   #只剩单词的训练集
test["clean_comment_text"] = clean_text(test['comment_text'])     #只剩单词的测试集

all_comment_list = list(train['clean_comment_text']) + list(test['clean_comment_text']) #将训练集与测试集的评论文本放在一个列表里
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', token_pattern=r'\w{1,}',
                         max_features=5000, ngram_range=(1, 1), analyzer='word')
#对训练集和测试集所有评论提取指定个数特征
text_vector.fit(all_comment_list)

#特征词向量及对应的概率
train_vec = text_vector.transform(train['clean_comment_text'])
test_vec = text_vector.transform(test['clean_comment_text'])


#将训练集随机分为训练数据和验证数据(x_train训练集数据，x_valid验证集数据，y_train训练集类别，y_valid验证集类别，)
x_train, x_valid, y_train, y_valid = train_test_split(train_vec, train[labels], test_size=0.1, random_state=2018)
x_test = test_vec

# #朴素贝叶斯
# for label in labels:
#     clf = MultinomialNB(alpha=0.001).fit(x_train, y_train[label])
#     y_pre = clf.predict(x_valid)
#
#     train_scores = clf.score(x_train, y_train[label])   #模型对于训练集的得分
#     valid_scores = accuracy_score(y_pre, y_valid[label])   #准确率得分
#     print(valid_scores)
#     exit()


#逻辑回归
accuracy = []
for label in labels:
    # clf = LogisticRegression(C=6).fit(x_train, y_train[label])  #逻辑回归
    clf = MultinomialNB(alpha=0.000001).fit(x_train, y_train[label])   #朴素贝叶斯
    y_pre = clf.predict(x_valid)   #返回预测标签（一个分类有多个标签，此处是0和1）

    train_scores = clf.score(x_train, y_train[label])   #模型对于训练集的得分
    valid_scores = accuracy_score(y_pre, y_valid[label])   #准确率得分
    print("{} train score is {}, valid score is {}".format(label, train_scores, valid_scores))
    accuracy.append(valid_scores)
    #predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率，所以每一行的和应该等于1
    pred_proba = clf.predict_proba(x_test)[:, 1]   #返回预测属于某标签的概率，[:, 1]表示标签等于1的概率
    sample_submission[label] = pred_proba
    sample_submission.to_csv('data1.csv', columns=["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], index=False)


print("Total cv accuracy is {}".format(np.mean(accuracy)))