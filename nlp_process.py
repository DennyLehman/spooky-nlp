# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:24:25 2019

@author: slin2
"""

# ideas from the following kernel
# https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial/notebook
import pandas as pd
import os
import matplotlib.pyplot as plt
print('hello world')

# load data
path = os.getcwd()
train = pd.read_csv(os.path.join(path,'data','train.csv'))
col = train.columns

a_count = train.groupby('author').count()['id'].reset_index()


plt.bar(a_count['author'],a_count['id'],color=['darkred','blue','green'],label=['Edgar','HP','Mary'])

all_words = train['text'].str.split(expand=True).unstack().value_counts()

i = 20
# this mess is the way to get ordered values in pyplot
# https://stackoverflow.com/questions/47373762/pyplot-sorting-y-values-automatically
plt.bar(range(i),all_words.values[0:i])
plt.xticks(range(i),all_words.index[0:i])
plt.ylabel('count')
plt.xlabel('word')
plt.title('the first {} words in ranking'.format(i))
plt.show()


# https://www.nltk.org/
import nltk
# https://stackoverflow.com/questions/26570944/resource-utokenizers-punkt-english-pickle-not-found
nltk.download('punkt')

# test import
nltk.word_tokenize('this is sample text')

# tokenization
# tokenization splits the sentence into individual words. Stronger packes exist to 
# process things like tweets or initials (E.D. Lehman wont be separate sentences)
first_sentence = nltk.word_tokenize(train.loc[0,'text'])



# stop words
# https://www.nltk.org/nltk_data/
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('English')

# stemming

# Vectorization