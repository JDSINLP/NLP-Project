import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from math import sqrt
from sklearn.metrics import mean_squared_error

import re
import unicodedata
import nltk

from wordcloud import WordCloud


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
    
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test


def prep_curriculum(df):
    
    '''
    This function is used to clean and prepare the curriculum logs data for manipulation 
    '''
    
    df = df[df['program_id']!=4]

    df.drop(columns=['Unnamed: 0'], inplace=True)
    
    df = df[df['path']!='/']
    
    return df



def mm_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Min Max Scaler and out puts
    scaled data for all three data splits
    '''
    # calls the Min Max Scaler function and fits to train data
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = mm_scaler.transform(train[col_list])
    val[col_list] = mm_scaler.transform(val[col_list])
    test[col_list] = mm_scaler.transform(test[col_list])
    
    return train, val, test

def ss_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Standard Scaler and out puts
    scaled data for all three data splits
    '''

    # calls Standard Scaler function and fits to train data
    ss_scale = StandardScaler()
    ss_scale.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = ss_scale.transform(train[col_list])
    val[col_list] = ss_scale.transform(val[col_list])
    test[col_list] = ss_scale.transform(test[col_list])
    
    return train, val, test

def rs_scaler(train, val, test, col_list):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Robust Scaler and out puts
    scaled data for all three data splits
    '''

    # calls Robust Scaler funtion and fits to train data set
    rs_scale = RobustScaler()
    rs_scale.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = rs_scale.transform(train[col_list])
    val[col_list] = rs_scale.transform(val[col_list])
    test[col_list] = rs_scale.transform(test[col_list])
    
    return train, val, test

def qt_scaler(train, val, test, col_list, dist='normal'):
    
    '''
    Takes train, val, test data splits and the column list to train on. Fits to the Quantile Transformer and out puts
    scaled data for all three data splits
    '''

    # calls Quantile Transformer function and fits to train data set
    qt_scale = QuantileTransformer(output_distribution=dist, random_state=42)
    qt_scale.fit(train[col_list])
    
    # transforms all three data sets
    train[col_list] = qt_scale.transform(train[col_list])
    val[col_list] = qt_scale.transform(val[col_list])
    test[col_list] = qt_scale.transform(test[col_list])
    
    return train, val, test


def remove_outliers(df, num=8, k=1.5):

    '''
    This function is to remove the data above the upper fence and below the lower fence for each column.
    This removes all data deemed as an outlier and returns more accurate data. It ignores columns that 
    are categorical and only removes data for continuous columns.
    '''
    a=[]
    b=[]
    fences=[a, b]
    features= []
    col_list = []
    i=0
    for col in df:
            new_df=np.where(df[col].nunique()>num, True, False)
            if new_df:
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':

                    # for each feature find the first and third quartile
                    q1, q3 = df[col].quantile([.25, .75])

                    # calculate inter quartile range
                    iqr = q3 - q1

                    # calculate the upper and lower fence
                    upper_fence = q3 + (k * iqr)
                    lower_fence = q1 - (k * iqr)

                    # appending the upper and lower fences to lists
                    a.append(upper_fence)
                    b.append(lower_fence)

                    # appending the feature names to a list
                    features.append(col)

                    # assigning the fences and feature names to a dataframe
                    var_fences= pd.DataFrame(fences, columns=features, index=['upper_fence', 'lower_fence'])
                    
                    col_list.append(col)
                else:
                    print(f'{col} is not a float or int')
            else:
                print(f'{col} column ignored')

    # for loop used to remove the data deemed unecessary 
    for col in col_list:
        df = df[(df[col]<= a[i]) & (df[col]>= b[i])]
        i+=1
    return df, var_fences


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def rmse(preds, target):
    return sqrt(mean_squared_error(preds['actual'], preds[target]))


def select_kbest(df, cont, cat, y, k):
    
    '''
    This function takes a data frame, a list of continuous variables, a list of categorical variables,
    the target variable, and top number of features wanted. It scales the continuous variables and 
    creates X_train and y_train data frames. It then creates dummies for the categorical variables. 
    After all the data has been manipulated it runs the SelectKBest for f_regression and returns 
    the top k number of variables.
    '''
    
    # fitting and scaling the continuous variables
    mms = MinMaxScaler()
    df[cont] = mms.fit_transform(df[cont])
    
    # creating X_train and y_train data frames
    X_df_scaled = df.drop(columns=[y])
    y_df = df[y]
    
    # creating dummies for the categorical variables
    X_df_scaled = pd.get_dummies(X_df_scaled, columns=cat)
    
    # fitting the regression model to the data
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(X_df_scaled, y_df)
    
    # determining which variables are the top k variables
    f_select_mask = f_selector.get_support()
    
    # returning data frame of the only the top k variables
    return X_df_scaled.iloc[:,f_select_mask]


def rfe(df, cont, cat, y, k):
    
    '''
    This function takes a data frame, a list of continuous variables, a list of categorical variables,
    the target variable, and top number of features wanted. It scales the continuous variables and 
    creates X_train and y_train data frames. It then creates dummies for the categorical variables.
    The function then runs the RFE function using linear regression to determine which features are best.
    It returns a data frame with each features and the ranking for the user to determine which features
    they want to use.
    '''
    
    # fitting and scaling the continuous variables
    mms = MinMaxScaler()
    df[cont] = mms.fit_transform(df[cont])
    
    # creating X_train and y_train data frames
    X_df_scaled = df.drop(columns=[y])
    y_df = df[y]
    
    # creating dummies for the categorical variables
    X_df_scaled = pd.get_dummies(X_df_scaled, columns=cat)
        
    # creating linear regressiong RFE model based on k number
    lm = LinearRegression()
    model = RFE(lm, n_features_to_select=k)
    
    # fitting model to scaled data
    model.fit(X_df_scaled, y_df)
    
    # determine rankings for each feature
    ranks = model.ranking_
    columns = X_df_scaled.columns.tolist()
    
    # creating data frame of ranking and column names
    feature_ranks = pd.DataFrame({'ranking':ranks,
                                  'feature':columns})
    
    # returns created data frame of feature rankings
    return feature_ranks.sort_values('ranking')
    

def clean_data(string):
    
    string = string.lower()
    
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
    
    string = re.sub(r'[^a-z0-9\s]', '', string)
    
    return string

def tokenize(string):
    
    tokenize = nltk.tokenize.ToktokTokenizer()
    
    tokens = tokenize.tokenize(string)
    
    return tokens

def stem(tokens):
    
    ps = nltk.porter.PorterStemmer()
    
    ps.stem('calling'), ps.stem('calls'), ps.stem('called'), ps.stem('call')
    ps.stem('house'), ps.stem('housing')
    
    stems = [ps.stem(word) for word in tokens]
    
    return ' '.join(stems)

def lemmatize(tokens):
    
    wnl = nltk.stem.WordNetLemmatizer()
    
    wnl.lemmatize('calling'), wnl.lemmatize('calls'), wnl.lemmatize('called'), wnl.lemmatize('call')
    wnl.lemmatize('house'), wnl.lemmatize('housing')
    wnl.lemmatize('mouse'), wnl.lemmatize('mice')
    
    lemmas = [wnl.lemmatize(word) for word in tokens]
    
    return ' '.join(lemmas)


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    
    stopwords_english = stopwords.words('english')
    
    stopwords_english.extend(extra_words)
    stopwords_english = [word for word in stopwords_english if word not in exclude_words]
    
    string_with_stopwords_removed = [word for word in string if word not in stopwords_english]
    
    return ' '.join(string_with_stopwords_removed)

def clean_text(text, extra_stopwords=['r', 'u', '2', 'ltgt']):
    
    wnl = nltk.stem.WordNetLemmatizer()
    
    stopwords = nltk.corpus.stopwords.words('english') + extra_stopwords
    
    clean_text = (unicodedata.normalize('NFKD', text)
                   .encode('ascii', 'ignore')
                   .decode('utf-8', 'ignore')
                   .lower())
    
    words = re.sub(r'[^\w\s]', '', clean_text).split()
    
    return [wnl.lemmatize(word) for word in words if word not in stopwords]