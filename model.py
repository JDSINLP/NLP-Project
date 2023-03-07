
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def train_val_test(df, target=None, stratify=None, seed=42):
    
    '''Split data into train, validate, and test subsets with 60/20/20 ratio'''
    
    train, val_test = train_test_split(df, train_size=0.6, random_state=seed)
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    return train, val, test


def get_tfid(X_train, X_val, X_test, y_train, y_val, y_test):
    
    tfidf = TfidfVectorizer()
    X_train_tfid = tfidf.fit_transform(X_train)
    X_val_tfid = tfidf.transform(X_val)
    X_test_tfid = tfidf.transform(X_test)
    return X_train_tfid, X_val_tfid, X_test_tfid


def get_baseline_accuracy( y_train):
    '''get baseline accuracy score'''
    
    # assign most common class to baseline
    baseline = y_train.mode()
    
    # compare baseline with y_train class to get most common class
    matches_baseline_prediction = (y_train == 'Python')
    
    # get mean
    baseline_accuracy = matches_baseline_prediction.mean()
    
    # print baseline accuracy
    print(f"Baseline accuracy: {round(baseline_accuracy, 2) * 100} %")


def decision_tree(x_trains, x_validates, y_train, y_validate):
    
    '''takes in x_train, x_validate, y_train, y_validate dataframes and returns a dataframe with accuracy score on train and validate data and the accuracy difference ''' 
    
    # create an empty list to append output
    metrics = []

    for i in range(1,10):
    # create model
        clf = DecisionTreeClassifier(max_depth=i, random_state=42)

        # fit the model to training data
        clf.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = clf.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate = clf.score(x_validates,y_validate)
        
        
        output = {'max_depth': i,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        
        metrics.append(output)
       
    # create a dataframe
    df = pd.DataFrame(metrics)
    
    # create a new column for a dataframe with a differance of train accuracy score and validate accuracy score
    df['difference'] = df.train_accuracy - df.validate_accuracy
    
    return df


def random_forest_tree(x_trains, x_validates, y_train, y_validate):
    '''takes in x_train, x_validate, y_train, y_validate dataframes
     and returns a dataframe with accuracy score on train and validate data and the accuracy difference ''' 
        
    # create an empty list to append output
    metrics = []
    
    for i in range(1, 25):
    
        # create model
        rf = RandomForestClassifier(min_samples_leaf =i, random_state=42) 

        # fit the model to training data
        rf.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = rf.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate = rf.score(x_validates,y_validate)

        output = {'min_samples_leaf ': i,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        
        metrics.append(output)
    
    # create a dataframe
    df = pd.DataFrame(metrics)
        
    # create a new column for a dataframe with a differance of train accuracy score and validate accuracy score
    df['difference'] = df.train_accuracy - df.validate_accuracy
    
    return df


def knn(x_trains, x_validates, y_train, y_validate):
    '''takes in x_train, x_validate, y_train, y_validate dataframes
    and returns a dataframe with accuracy score on train and validate dataand the accuracy difference ''' 
    
    # create an empty list to append output
    metrics = []
    
    for i in range(1,15):

        # create model
        knn = KNeighborsClassifier(n_neighbors=i) 

        # fit the model to training data
        knn.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = knn.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate = knn.score(x_validates,y_validate)

        output = {'n_neighbors': i,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        
        metrics.append(output)
        
    # create a dataframe
    df = pd.DataFrame(metrics)
    
    # create a new column for a dataframe with a differance of train accuracy score and validate accuracy score
    df['difference'] = df.train_accuracy - df.validate_accuracy
    
    return df


def get_decision_tree(x_trains, x_validates, y_train, y_validate, n):
    '''get decision tree accuracy score on train and validate data'''
    
    # create model
    clf = DecisionTreeClassifier(max_depth = n, random_state=42)

    # fit the model to train data
    clf.fit(x_trains, y_train)

    # compute accuracy
    train_acc = clf.score(x_trains, y_train)
    validate_acc = clf.score(x_validates, y_validate)

    return train_acc, validate_acc


def get_random_forest(x_trains, x_validates, y_train, y_validate, n):
    '''get random forest accuracy score on train and validate data'''
    
    # create model
    rf= RandomForestClassifier(min_samples_leaf = n, random_state=42) 

    # fit the model to train data
    rf.fit(x_trains, y_train)

    # compute accuracy
    train_acc = rf.score(x_trains, y_train)
    validate_acc = rf.score(x_validates, y_validate)

    return train_acc, validate_acc


def get_knn(x_trains, x_validates, y_train, y_validate, n):
    ''' get KNN accuracy score on train and validate data'''
    
    # create model
    knn= KNeighborsClassifier(n_neighbors = n) 

    # fit the model to train data
    knn.fit(x_trains, y_train)

    # compute accuracy
    train_acc = knn.score(x_trains, y_train)
    validate_acc = knn.score(x_validates, y_validate)
    
    return train_acc, validate_acc


def get_models_accuracy(X_train, X_val, y_train, y_val):
    '''takes x_trains, y_train, x_validates, y_validate, train, validate, target
    return dataframe with models and their accuracy score on train and validate data
    '''
    # get accuracy
#     baseline_accuracy = mo.get_baseline_accuracy( y_train)
    tree_train_acc, tree_validate_acc= get_decision_tree(X_train, X_val, y_train, y_val, 4)
    random_train_acc, random_validate_acc= get_random_forest(X_train, X_val, y_train, y_val, 7)
    knn_train_acc, knn_validate_acc= get_knn(X_train, X_val, y_train, y_val, 4)    
    # assing index
    index = ['Decision_Tree', 'Random_Forest', 'KNN']
    
    # create a dataframe
    df = pd.DataFrame({
                       'train_accuracy':[tree_train_acc, random_train_acc, knn_train_acc],
                       'validate_accuracy': [tree_validate_acc, random_validate_acc, knn_validate_acc]},
                          index=index)
    df['difference']= df['train_accuracy']-df['validate_accuracy']
    
    return df


def viz_models_accuracy(df):
   
    '''takes in a dataframe and plot a graph to show comparisons models accuracy score on train and valiadate data'''
    
    df_1 = df.copy()
    df_1.validate_accuracy = df_1.validate_accuracy * 100
    df_1.train_accuracy = df_1.train_accuracy * 100
    df_1 = df_1.drop(columns='difference')
    df_1 = df_1.sort_values(by=['validate_accuracy'], ascending=False)
    ax = df_1.plot.bar(rot=.5)
    
  
    baseline_accuracy = 39
    plt.axhline(baseline_accuracy , label="Baseline Accuracy", color='red')
    plt.legend()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Comparisons of Accuracy")
    plt.ylabel('Accuracy score')
    plt.bar_label(ax.containers[0],fmt='%.0f%%')
    plt.bar_label(ax.containers[1],fmt='%.0f%%')
    sns.set_theme(style="whitegrid")
    plt.show()


def get_decison_tree_test(x_train, x_test, y_train, y_test,n):
    ''' get decision tree accuracy score on test'''
   
    clf = DecisionTreeClassifier(max_depth=n, random_state=42)
    
    clf.fit(x_train, y_train)
    
    validate_acc = clf.score(x_test, y_test)
    
    print(f"validate Accuracy: {round(validate_acc, 2) * 100} %")
