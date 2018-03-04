#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#
# print(r"\n")
text = "Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27"
text = re.sub(r"[a-zA-Z]*'[a-zA-Z]*", "", text)
text_vector = TfidfVectorizer()
text_vector.fit(text)
train_vec = text_vector.transform(text)
# text = re.sub(r"[^A-Za-z]", " ", text)  #正则替换
print(train_vec)