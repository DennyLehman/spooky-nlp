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

new_word_list = [word for word in first_sentence if word.lower() not in stop_words]
print(first_sentence)
print(new_word_list)
print(len(first_sentence))
print('{} words were removed from the first sentence with stop word removal'.format(len(first_sentence)-len(new_word_list)))

# stemming
# gets the root of the words
stemmer = nltk.stem.PorterStemmer()
print("The stemmed form of running is: {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))

print("The stemmed form of leaves is: {}".format(stemmer.stem("leaves")))

# Lemminization
# https://stackoverflow.com/questions/13965823/resource-corpora-wordnet-not-found-on-heroku
#'''
#NLTK Stemmers
#
#Interfaces used to remove morphological affixes from words, leaving only the word stem. Stemming algorithms aim to remove those affixes required for eg. grammatical role, tense, derivational morphology leaving only the stem of the word. 
#'''
nltk.download('wordnet')
lemm = nltk.stem.WordNetLemmatizer()
lemm.lemmatize('leaves')

# Vectorization
# turns sentences into vectors of word counts

import sklearn

