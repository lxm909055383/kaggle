#!/usr/bin/python
#coding:utf-8

import pandas as pd
import numpy as np
import re

text = 'ab09@_??&'
text = re.sub(r"[^A-Za-z]", " ", text)  #正则替换
print(text)