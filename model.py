
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




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



def get_logistic_regression(x_trains, x_validates, y_train, y_validate, n):
    '''get logistic regrssion accuracy score on train and validate data'''
    
    # create model
    logit = LogisticRegression(C = n, random_state=42, solver='liblinear')

    # fit the model to train data
    logit.fit(x_trains, y_train)

    # compute accuracy
    train_acc = logit.score(x_trains, y_train)
    validate_acc = logit.score(x_validates, y_validate)
    
    return train_acc, validate_acc



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





def logistic_regression(x_trains, x_validates, y_train, y_validate):
    # using Logistic regression model with different values of hyperparameter c to find best model

    # create an empty list to append output
    metrics = []

    # create model1 of logistic regression
    logit1 = LogisticRegression(C = 1, random_state=42, solver='liblinear')
    logit2 = LogisticRegression(C = 0.1, random_state=42, solver='liblinear')

    cols = [logit1, logit2]

    for col in cols : 

        # fit model
        col.fit(x_trains, y_train)

        # fit the model to training data
        col.fit(x_trains, y_train)

        # accuracy score on train
        accuracy_train = col.score(x_trains,y_train)

        # accuracy score on validate
        accuracy_validate =col.score(x_validates,y_validate)

        output = {'model': col,
                 'train_accuracy': accuracy_train,
                 'validate_accuracy': accuracy_validate,
                 }
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    
    df['difference'] = df.train_accuracy - df.validate_accuracy
    
    return df




def get_models_accuracy(X_train, X_val, y_train, y_val):
    '''takes x_trains, y_train, x_validates, y_validate, train, validate, target
    return dataframe with models and their accuracy score on train and validate data
    '''
    # get accuracy
#     baseline_accuracy = mo.get_baseline_accuracy( y_train)
    tree_train_acc, tree_validate_acc= get_decision_tree(X_train, X_val, y_train, y_val, 4)
    random_train_acc, random_validate_acc= get_random_forest(X_train, X_val, y_train, y_val, 7)
    logistic_train_acc, logistic_validate_acc = get_logistic_regression(X_train, X_val, y_train, y_val, 1)
    
    # assing index
    index = ['Decision_Tree(max_depth=4)', 'Random_Forest(min_samples_lead=7)', 'Logistic_Regression(C=1)']
    
    # create a dataframe
    df = pd.DataFrame({'train_accuracy':[tree_train_acc, random_train_acc, logistic_train_acc],
                       'validate_accuracy': [tree_validate_acc, random_validate_acc,logistic_validate_acc]},
                          index=index)
    df['difference']= df['train_accuracy']-df['validate_accuracy']
    
    return df

def viz_models_accuracy(df):
    '''takes in a dataframe and plot a graph to show comparisons models accuracy score on train and valiadate data'''
    df.train_accuracy = df.train_accuracy * 100
    df.validate_accuracy = df.validate_accuracy * 100
#     plt.figure(figsize=(3,6))
    ax = df.drop(columns='difference').plot.bar(rot=75)
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Comparisons of Accuracy")
    plt.ylabel('Accuracy score')
#     ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.bar_label(ax.containers[0],fmt='%.0f%%')
    plt.bar_label(ax.containers[1],fmt='%.0f%%')
    

    plt.show()


def get_decison_tree_test(x_train, x_test, y_train, y_test,n):
    ''' get decision tree accuracy score on test'''
   
    clf = DecisionTreeClassifier(max_depth=n, random_state=42)
    
    clf.fit(x_train, y_train)
    
    validate_acc = clf.score(x_test, y_test)
        
    print(validate_acc)




def get_models_accuracy(X_train, X_val, y_train, y_val):
    '''takes x_trains, y_train, x_validates, y_validate, train, validate, target
    return dataframe with models and their accuracy score on train and validate data
    '''
    # get accuracy
#     baseline_accuracy = mo.get_baseline_accuracy( y_train)
    tree_train_acc, tree_validate_acc= get_decision_tree(X_train, X_val, y_train, y_val, 4)
    random_train_acc, random_validate_acc= get_random_forest(X_train, X_val, y_train, y_val, 7)
    logistic_train_acc, logistic_validate_acc = get_logistic_regression(X_train, X_val, y_train, y_val, 1)
    
    # assing index
    index = ['Decision_Tree(max_depth=4)', 'Random_Forest(min_samples_lead=7)', 'Logistic_Regression(C=1)']
    
    # create a dataframe
    df = pd.DataFrame({'train_accuracy':[tree_train_acc, random_train_acc, logistic_train_acc],
                       'validate_accuracy': [tree_validate_acc, random_validate_acc,logistic_validate_acc]},
                          index=index)
    df['difference']= df['train_accuracy']-df['validate_accuracy']
    
    return df



def viz_models_accuracy(df):
    '''takes in a dataframe and plot a graph to show comparisons models accuracy score on train and valiadate data'''
    df.train_accuracy = df.train_accuracy * 100
    df.validate_accuracy = df.validate_accuracy * 100
#     plt.figure(figsize=(3,6))
    ax = df.drop(columns='difference').plot.bar(rot=75)
    ax.spines[['right', 'top']].set_visible(False)
    plt.title("Comparisons of Accuracy")
    plt.ylabel('Accuracy score')
#     ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    plt.bar_label(ax.containers[0],fmt='%.0f%%')
    plt.bar_label(ax.containers[1],fmt='%.0f%%')
    

    plt.show()