#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:08:07 2021

@author: Alexandre Bovet
"""


from TwSentiment import CustomTweetTokenizer, bag_of_words_and_bigrams
import pandas as pd

from zipfile import ZipFile

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import pickle

import numpy as np

raise Exception
#%% load training set
with ZipFile('trainingandtestdata.zip', mode='r') as fopen:
    
    df_train =  pd.read_csv(fopen.open('training.1600000.processed.noemoticon.csv'),
                            encoding='latin1', header=None)
    
df_train.columns = ['polarity', 'id','date','query','user','text']    
    
tokenizer = CustomTweetTokenizer(preserve_case=False, # keep Upper cases
                                 reduce_len=True, # reduce repetition of letter to a maximum of three
                                 strip_handles=False, # remove usernames (@mentions)
                                 normalize_usernames=True, # replace all mentions to "@USER"
                                 normalize_urls=True, # replace all urls to "URL"
                                 keep_allupper=True) # keep upercase for words that are all in uppercase


#%% features vect

features = [bag_of_words_and_bigrams(tokenizer.tokenize(t)) for t in df_train.text.tolist()]

labels = df_train.polarity.tolist()
y = np.array([1 if l > 0 else 0 for l in labels])

vect = DictVectorizer(dtype=np.int8, sparse=True, sort=False)
X = vect.fit_transform(features)

# memmaping of the features
joblib.dump(X, '_features_vect.memmap')
joblib.dump(y, '_labels_vect.memmap')

#%% cross-val
X = joblib.load('_features_vect.memmap')
y = joblib.load('_labels_vect.memmap')

scoring = 'f1_micro'
n_splits = 10
loss = 'log'
penalty = 'l2'
grid_search_parameters = {'classifier__alpha' : np.logspace(-1,-7, num=20)}  

pipeline_list = [('classifier', SGDClassifier(verbose=True, 
                                                     loss=loss,
                                                     penalty=penalty))]
pipeline = Pipeline(pipeline_list)
    
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=34)
        
grid_search = GridSearchCV(estimator=pipeline, param_grid=grid_search_parameters, 
                           cv=kfold,
                           scoring=scoring, 
                           verbose=1 , 
                           n_jobs=4)

grid_search.fit(X, y)

alpha = grid_search.best_estimator_.get_params()['classifier__alpha']

#%% train classifier with the best parameters

pipeline_list = [('feat_vectorizer', DictVectorizer(dtype=np.int8, sparse=True, sort=False)),
                         ('classifier', SGDClassifier(loss=loss,
                                   alpha=alpha,
                                   penalty=penalty,
                                   random_state=42))]

pipeline = Pipeline(pipeline_list)

pipeline.fit(features, y)

#%% save classifier

with open('tweet_classifier_pipepline.pickle', 'wb') as fopen:
    pickle.dump(pipeline, fopen)
