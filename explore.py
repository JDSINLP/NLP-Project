import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import re
import unicodedata
import pandas as pd

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import seaborn as sns
import json
# import acquire
import prepare
import wrangle as w

from env import github_token, github_username
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


df=pd.read_json('data2.json')

#Question 1
def viz_most_common_unigrams(words):
    '''takes in words, get top 20 unigram words, plot bar graph of top 20 unigram words'''
    
    words_unigrams = pd.Series(nltk.ngrams(words.split(), 1))
    top_20_words = words_unigrams.value_counts().head(20)
    top_20_words.sort_values().plot.barh(color='pink', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring words unigrams')
    plt.ylabel('unigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_words.reset_index()['index'].apply(lambda t: t[0])
    _ = plt.yticks(ticks, labels)
    

def viz_most_common_bigrams(words):
    '''takes in words, get top 20 bigram words, plot bar graph of top 20 bigram words'''
    
    words_bigrams = pd.Series(nltk.ngrams(words.split(), 2))
    top_20_words = words_bigrams.value_counts().head(20)
    top_20_words.sort_values().plot.barh(color='pink', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring words bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_words.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
def viz_most_common_trigrams(words):
    '''takes in words, get top 20 bigram words, plot bar graph of top 20 bigram words'''
    
    words_trigrams = pd.Series(nltk.ngrams(words.split(), 3))
    top_20_words = words_trigrams.value_counts().head(20)
    top_20_words.sort_values().plot.barh(color='pink', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring words trigrams')
    plt.ylabel('Trigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_words.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1] + ' ' + t[2])
    _ = plt.yticks(ticks, labels)

def Q1(train):
    words = w.clean_text(' '.join(train['readme_contents_clean']))
    viz_most_common_unigrams(words)
    plt.show()
    viz_most_common_bigrams(words)
    plt.show()
    viz_most_common_trigrams(words)
    plt.show()
    
    
#Question 2 
def viz_length_content(train):
    '''takes in a dataframe, plot a bar graph of a lenth of content by programming language'''
    
    ax = sns.barplot(data=train, x='language',y='length' ,order=[ 'C++','Python' ,'Java', 'JavaScript'])
    avg_length = train.length.mean()
    plt.axhline(avg_length , label="Avg Length", color='yellow')
    plt.legend()
    plt.xlabel('Programming Language')
    plt.ylabel('Length')
    plt.ylim(0,12000)
    plt.title('Variation of length of README by programming language')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()

#Question 3   
def viz_count_unique(train):
    '''takes in a dataframe, plot a bar graph of a number of unique words of content by programming language'''

    
    ax = sns.barplot(data=train, x='unique',y='language' ,orient ='h', order=[  'Java','Python','JavaScript','C++'])
    plt.ylabel('Programming Language')
    plt.xlabel('Length')
#     plt.xlim(30,40)
    plt.title('Variation of number of unique words by programming language')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()


