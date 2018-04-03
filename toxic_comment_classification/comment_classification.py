#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics



#读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  #六个分类

#统计每个标签0和1的个数
countdata = train.iloc[:, 2:].apply(pd.Series.value_counts)
print(countdata)

#检查是否有缺失值
null_check = train.isnull().sum()
print(null_check)

#同时有多个标签的样本数
rowsums = train.iloc[:, 2:].sum(axis=1)
x = rowsums.value_counts()
print(x)

# #数据预处理
# def clean_text(comment_text):
#     comment_list = []
#     for text in comment_text:
#         # 将单词转换为小写
#         text = text.lower()
#         #将缩写词去掉
#         text = re.sub(r"[a-z]*'[a-z]*", "", text)
#         # 将非字母正则替换为空格
#         text = re.sub(r"[^a-z]", " ", text)
#         #添加到评论列表
#         comment_list.append(text)
#     return comment_list
#
# train["clean_comment_text"] = clean_text(train['comment_text'])   #只剩单词的训练集
# test["clean_comment_text"] = clean_text(test['comment_text'])     #只剩单词的测试集
# all_comment_list = list(train['clean_comment_text']) + list(test['clean_comment_text']) #将训练集与测试集的评论文本放在一个列表里
#
# #使用CountVectorizer,转化为词频矩阵
# text_vector = CountVectorizer()
#
# #使用TfidfVectorizer,转化为词语的tf-idf频率
# # text_vector = TfidfVectorizer()
#
# #使用CountVectorizer,并去掉停用词
# # text_vector = CountVectorizer(analyzer='word', stop_words='english')
#
# #使用TfidfVectorizer,并去掉停用词
# # text_vector = TfidfVectorizer(analyzer='word', stop_words='english')
#
# #使用TfidfVectorizer,并加入其它参数
# # text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', token_pattern=r'\w{1,}', max_features=500, ngram_range=(1, 1), analyzer='word')
#
# #对训练集和测试集所有评论提取指定个数特征
# text_vector.fit(all_comment_list)
# #特征词向量及对应的概率
# train_vec = text_vector.transform(train['clean_comment_text'])
# test_vec = text_vector.transform(test['clean_comment_text'])
#
#
# #将训练集随机分为训练数据和验证数据(x_train训练集数据，x_valid验证集数据，y_train训练集类别，y_valid验证集类别，)
# x_train, x_valid, y_train, y_valid = train_test_split(train_vec, train[labels], test_size=0.1, random_state=201)
# x_test = test_vec
#
# def metrics_result(actual, predict):
#     print('精确率:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
#     print('召回率:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
#     print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
#
# accuracy = []
# for label in labels:
#     # clf = LogisticRegression(C=6).fit(x_train, y_train[label])  #逻辑回归训练参数
#     clf = MultinomialNB(alpha=0.000001).fit(x_train, y_train[label])   #朴素贝叶斯训练参数
#     y_pre = clf.predict(x_valid)  # 预测验证集标签（一个分类有多个标签，此处是0和1）
#
#     #自带得分函数计算模型的准确率
#     model_score = clf.score(x_valid, y_valid[label])
#     accuracy.append(model_score)
#     #计算模型的准确率(自带得分函数)、精确率、召回率、f1-score
#     print('------{}的分类精度------'.format(label))
#     print('准确率:{0:.3f}'.format(model_score))
#     metrics_result(y_valid[label], y_pre)
#
#     #predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率，所以每一行的和应该等于1
#     # pred_proba = clf.predict_proba(x_test)[:, 1]   #返回预测属于某标签的概率，[:, 1]表示标签等于1的概率
#     # sample_submission[label] = pred_proba
#     # sample_submission.to_csv('data1.csv', columns=["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], index=False)
#
# print("模型总的准确率为：{0:.3f}".format(np.mean(accuracy)))




