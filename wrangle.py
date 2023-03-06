# imports
import pandas as pd

import unicodedata
import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def bin_language(df):
    '''takes in a dataframe and bin language and return a dataframe'''
    
    for i, row in df.iterrows():
        if str(row['language']) in ['Python', 'JavaScript', 'C++', 'Java']:
            continue
        else:
            df.iloc[i]['language'] = 'Other'
    return df


def clean_text(text, extra_stop_words=[],exclude_stop_words=[]):
    '''takes in a dataframe, list of extra stop words, list of words to exclude stop words
    clean text using lemmatization, lower string into lowercase, remove that is not alphabet, digits
    remove words that are not in stopwords
    rerun clean words'''
    
    wnl = nltk.stem.WordNetLemmatizer()
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    clean_text = (unicodedata.normalize('NFKD', text)
                   .encode('ascii', 'ignore')
                   .decode('utf-8', 'ignore')
                   .lower())
    
    words = re.sub(r'[^\w\s]', ' ', clean_text).split()
    
    clean_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return ' '.join(clean_words)


def prepare_df(df):
    '''takes in a dataframe
    add columns with clean data, lenght of data, number of unique words
    return a dataframe'''
    
    df = bin_language(df)
    df = df[~(df.language =='Other')].reset_index().drop(columns='index')
    
    df['readme_contents_clean'] = df['readme_contents'].apply(clean_text)
    
    df['length'] = df['readme_contents'].str.len()
    
    lists = []
    for i,row in df.iterrows():
        words = df.iloc[i]['readme_contents_clean']
        lists.append(len(set(words)))
        
    df['unique'] = pd.Series(lists)
    
    return df


def train_val_test(df, target=None, stratify=None, seed=42):
    
    '''Split data into train, validate, and test subsets with 60/20/20 ratio'''
    
    train, val_test = train_test_split(df, train_size=0.6, random_state=seed)
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    return train, val, test


def x_y_split(df, target, seed=42):
    
    '''
    This function is used to split train, val, test into X_train, y_train, X_val, y_val, X_train, y_test
    '''
    
    train, val, test = train_val_test(df, target, seed)
    
    X_train = train.readme_contents_clean
    y_train = train[target]

    X_val = val.readme_contents_clean
    y_val = val[target]

    X_test = test.readme_contents_clean
    y_test = test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test
