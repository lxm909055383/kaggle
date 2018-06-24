from textblob import TextBlob
import nltk

# from nltk.corpus import treebank
# t = treebank.parsed_sents('wsj_0001.mrg')[0]
# print(t.draw())

# nltk.download('averaged_perceptron_tagger')


wiki = TextBlob("Python is a high-level, general-purpose programming language.")
print(wiki.tags)