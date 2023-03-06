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
import dev_wrangle as w

from env import github_token, github_username
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


df=pd.read_json('data2.json')


#Question 1
def Q1(train):

    words = w.clean_text(' '.join(train['readme_contents_clean']))
    w.viz_most_common_unigrams(words)
    plt.show()
    w.viz_most_common_bigrams(words)
    plt.show()


#Question 2
def Q2(train):
    w.viz_length_content(train)


# Question 3
def Q3(train):
    train.groupby('language').unique.value_counts()
    w.viz_count_unique(train)
    plt.show()
    
