#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 07:27:13 2022

@author: arpanrajpurohit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer as suffix_stripper
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

DATABASE_PATH = "dataset/Sarcasm_Headlines_Dataset.json"
IGNORE_WORD_LIST = "stopwords"
ROW_IN_SHAPE = 0
DATABASE_COLUMN_HEADLINE_NAME = "headline"
DATABASE_OUTPUT_COLUMN = 2
LANGUAGE = 'english'
TEST_SIZE_IN_PER = 0.20
EXCLAMATION_IN_SYMBOL = '!'
EXCLAMATION_IN_WORD = ' exclamation'
QUESTION_IN_SYMBOL = '?'
QUESTION_IN_WORD   = ' question'
QUOTATION_IN_SYMBOL = r'\'(.+?)\''
QUOTATION_IN_WORD   = ' quotation'
NOT_LOWERCASE_LETTERS = '[^a-z]'
SPACE = ' '

FEATURE_START = 100
FEATURE_END   = 3000
FEATURE_STEP  = 100

LABEL_NUM_OF_MAX_VEC = 'Number of Max Vector'
LABEL_ERROR    = 'Error'
LABEL_ACCURACY = 'Accuracy'
LABEL_PRECISION = 'Precision'
LABEL_RECALL    = 'Recall'
#database import
database =  pd.read_json(DATABASE_PATH, lines = True)

#Data preprocessing
nltk.download(IGNORE_WORD_LIST)
from nltk.corpus import stopwords as sw
headlines = []
output = database.iloc[:, DATABASE_OUTPUT_COLUMN]
total_lines = database.shape[ROW_IN_SHAPE]

for index in range(0, total_lines):
    edit = re.sub(EXCLAMATION_IN_SYMBOL, EXCLAMATION_IN_WORD, database[DATABASE_COLUMN_HEADLINE_NAME][index])
    edit = edit.replace(QUESTION_IN_SYMBOL, QUESTION_IN_WORD)
    found = re.findall(QUOTATION_IN_SYMBOL, edit)
    if found:
        edit += QUOTATION_IN_WORD
    edit = re.sub(NOT_LOWERCASE_LETTERS, SPACE, edit)
    edit = edit.split()
    ss = suffix_stripper()
    edit = [ss.stem(word) for word in edit if not word in sw.words(LANGUAGE)]
    edit = SPACE.join(edit)
    headlines.append(edit)

#model implementation
features    = range(FEATURE_START, FEATURE_END, FEATURE_STEP)
error_rates = []
accuracies  = []
precisions  = []
recalls     = []

for feature in features:
    cv = CountVectorizer(max_features = feature)
    dbinput = cv.fit_transform(headlines).toarray()
    train_input, test_input, train_output, test_output = train_test_split(dbinput, output, test_size = TEST_SIZE_IN_PER, random_state = 0)
    model = GaussianNB()
    model.fit(train_input, train_output)
    test_predictions = model.predict(test_input)
    conf_mat = confusion_matrix(test_output, test_predictions)
    true_positive  = conf_mat[0][0]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    true_negative  = conf_mat[1][1]
    total_conf     = true_positive+true_negative+false_positive+false_negative
    
    error_rate = (false_positive+false_negative)/total_conf
    accuracy   = (true_positive + true_negative)/total_conf
    precision  = true_positive / (true_positive + false_positive)
    recall     = true_positive / (true_positive + false_negative)
    
    error_rates.append(error_rate)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    
optimal = features[error_rates.index(min(error_rates))]
print("the optimal numbers of max vectors is %d" % optimal + " with an error of %.2f" % min(error_rates) + " with an accuracy of %.2f" % max(accuracies))

plt.plot(features, error_rates)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_ERROR)
plt.show()

plt.plot(features, accuracies)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_ACCURACY)
plt.show()

plt.plot(features, precisions)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_PRECISION)
plt.show()

plt.plot(features, recalls)
plt.xlabel(LABEL_NUM_OF_MAX_VEC)
plt.ylabel(LABEL_RECALL)
plt.show()